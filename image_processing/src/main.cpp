// %HISTORY%
// March 25 2018 - Pawan Negi - Initialize
//
#include<iostream>
#include"face_detect/detect.hpp"
#include"face_recog/recog.hpp"
#include"face_track/track.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<mpi.h>

using namespace cv;
using namespace std;
int main(int argc, char** argv){

    cout << "this is main.cpp"<< endl ;
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }
    
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d"
           " out of %d processors\n",
           processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    detect::somedetectfucntion2(2);
    recog::somerecogfucntion2(2);
    track::sometrackfucntion2(2);
    
    MPI_Finalize();
    return 0;
}

