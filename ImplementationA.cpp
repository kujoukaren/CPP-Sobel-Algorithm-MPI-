#include "mpi.h"
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>

#define send_data_tag 2001
#define return_data_tag 2002


***************** Add/Change the functions(including processImage) here ********************* 

// NOTE: Some arrays are linearized in the skeleton below to ease the MPI Data Communication 
// i.e. A[x][y] become A[x*total_number_of_columns + y]
int* processImage(int* inputImage, int processId, int num_processes, int image_height, int image_width){
     int x, y, sum, sumx, sumy;
     int GX[3][3], GY[3][3];
     /* 3x3 Sobel masks. */
     GX[0][0] = -1; GX[0][1] = 0; GX[0][2] = 1;
     GX[1][0] = -2; GX[1][1] = 0; GX[1][2] = 2;
     GX[2][0] = -1; GX[2][1] = 0; GX[2][2] = 1;
     
     GY[0][0] =  1; GY[0][1] =  2; GY[0][2] =  1;
     GY[1][0] =  0; GY[1][1] =  0; GY[1][2] =  0;
     GY[2][0] = -1; GY[2][1] = -2; GY[2][2] = -1;
	
	//chunkSize is the number of rows to be computed by this process
	int chunkSize;
    int* partialOutputImage;
	
	for( x = start_of_chunk; x < end_of_chunk; x++ ){
		 for( y = 0; y < image_width; y++ ){
			 sumx = 0;
			 sumy = 0;
			
			//Change boundary cases as required
			 if(x==0 || x==(image_height-1) || y==0 || y==(image_width-1))
				 sum = 0;
			 else{

				 for(int i=-1; i<=1; i++)  {
					 for(int j=-1; j<=1; j++){
						 sumx += (inputImage[(x+i)*image_width+ (y+j)] * GX[i+1][j+1]);
					 }
				 }

				 for(int i=-1; i<=1; i++)  {
					 for(int j=-1; j<=1; j++){
						 sumy += (inputImage[(x+i)*image_width+ (y+j)] * GY[i+1][j+1]);
					 }
				 }

				 sum = (abs(sumx) + abs(sumy));
			 }
			 partialOutputImage[x*image_width + y] = (0 <= sum <= 255);
		 }
	}
	return partialOutputImage;
}

int main(int argc, char* argv[])
{
	int processId, num_processes, image_height, image_width, image_maxShades, root_process, child_id;
	int *inputImage, *outputImage;
	int *partialImage, *tempImage;
	int count_recv, count_return;
	
	root_process = 0;
	
	// Setup MPI
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &processId);
    MPI_Comm_size( MPI_COMM_WORLD, &num_processes);
	
	chunk_size = image_height / num_processes;
	
    if(argc != 3)
    {
		if(processId == 0)
			std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename>" << std::endl;
		MPI_Finalize();
        return 0;
    }
	
	if(processId == 0)
	{
		std::ifstream file(argv[1]);
		if(!file.is_open())
		{
			std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
			MPI_Finalize();
			return 0;
		}

		std::cout << "Detect edges in " << argv[1] << " using " << num_processes << " processes\n" << std::endl;

		std::string workString;
		/* Remove comments '#' and check image format */ 
		while(std::getline(file,workString))
		{
			if( workString.at(0) != '#' ){
				if( workString.at(1) != '2' ){
					std::cout << "Input image is not a valid PGM image" << std::endl;
					return 0;
				} else {
					break;
				}       
			} else {
				continue;
			}
		}
		/* Check image size */ 
		while(std::getline(file,workString))
		{
			if( workString.at(0) != '#' ){
				std::stringstream stream(workString);
				int n;
				stream >> n;
				image_width = n;
				stream >> n;
				image_height = n;
				break;
			} else {
				continue;
			}
		}
		inputImage = new int[image_height*image_width];
		outputImage = new int[image_height*image_width];
		tempImage = new int[image_height*image_width];
		partialImage = new int[image_height*image_width];

		/* Check image max shades */ 
		while(std::getline(file,workString))
		{
			if( workString.at(0) != '#' ){
				std::stringstream stream(workString);
				stream >> image_maxShades;
				break;
			} else {
				continue;
			}
		}
		/* Fill input image matrix */ 
		int pixel_val;
		for( int i = 0; i < image_height; i++ )
		{
			if( std::getline(file,workString) && workString.at(0) != '#' ){
				std::stringstream stream(workString);
				for( int j = 0; j < image_width; j++ ){
					if( !stream )
						break;
					stream >> pixel_val;
					inputImage[(i*image_width)+j] = pixel_val;
				}
			} else {
				continue;
			}
		}
	} // Done with reading image using process 0
	
	if(processId == 0){
		// distribute a portion to each child process
		for(child_id = 1; child_id < num_processes; child_id++) {
			start_of_chunk = child_id*chunk_size;
			end_of_chunk = (child_id + 1)*chunk_size - 1;
			
			int count = chunk_size * image_width;
			int start_index = child_id * count;
			
			// check last chuck's end point
			if((image_height- end_of_chunk) < chunk_size) {
				end_of_chunk = image_height - 1;
				int rows = end_of_chunk - start_of_chunk + 1;
				// overwrite count value
				count = rows * image_width;
			}
			
			// sending segments
			MPI_Send(&count, 1, MPI_INT, child_id, send_data_tag, MPI_COMM_WORLD);
			MPI_Send(&inputImage[start_index], count, MPI_INT, child_id, send_data_tag, MPI_COMM_WORLD);
		}
		
		// process image in the segment assigned to the root process
		for(int i = 0; i < (chunk_size * image_width); i++){
				tempImage[i] = inputImage[i];
		}
		outputImage = processImage(tempImage, processId, num_processes, image_height, image_width);
		
		// receive partial image and add them to outputImage
		for(child_id = 1; child_id < num_processes; child_id++) {
			MPI_Recv (&count_return, 1, MPI_INT, MPI_ANY_SOURCE, return_data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv (&partialImage, count_return, MPI_INT, MPI_ANY_SOURCE, return_data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			for(int i = 0; i < ;i++){
				outputImage[child_id*image_width + 1] = partialImage[i];
			}	
		}
	}
	else{
		// receive segments from root process
		MPI_Recv(&count_recv, 1, MPI_INT, root_process, send_data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		MPI_Recv(&tempImage, count_recv, MPI_INT, root_process, send_data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// process image in the segment assigned to each child process
		partialImage = processImage(tempImage, processId, num_processes, image_height, image_width);
		
		// send partial image to the root process
		MPI_Send(&count_recv, 1, MPI_INT, root_process, return_data_tag, MPI_COMM_WORLD);
		MPI_Send(&partialImage, count_recv, MPI_INT, root_process, return_data_tag, MPI_COMM_WORLD);
	}
	
	***************** Add code as per your requirement below ********************* 
	
	if(processId == 0)
	{
		/* Start writing output to your file */
		std::ofstream ofile(argv[2]);
		if( ofile.is_open() )
		{
			ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
			for( int i = 0; i < image_height; i++ )
			{
				for( int j = 0; j < image_width; j++ ){
					ofile << outputImage[(i*image_width)+j] << " ";
				}
				ofile << "\n";
			}
		} else {
			std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
			return 0;
		}	
	}
	 
    MPI_Finalize();
    return 0;
}