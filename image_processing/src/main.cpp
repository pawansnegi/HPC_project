// %HISTORY%
// March 25 2018 - Pawan Negi - Initialize
//
#include"face_detect/detect.hpp"
#include"face_recog/recog.hpp"
#include"face_track/track.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include<mpi.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace face;

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
        exit(1);
    }
    string output_folder = ".";
    if (argc == 3) {
        output_folder = string(argv[2]);
    }
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        recog::read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::FaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = images[11];
    int testLabel = labels[11];
    images.pop_back();
    labels.pop_back();
    
    int predictedLabel = recog::inbuilt_recognition(images , labels , testSample);
    string result_message = format("Predicted class = %d / Actual class = %d."
            , predictedLabel, testLabel);
    cout << result_message << endl ;
    
    
    for (int i = 0 ; i < images.size() ; i++){
        images[i] = images[i].reshape(1, 1 );
    }

    Mat mean = recog::calculate_mean(images) ;
    cout << mean.at<double>(112) << endl;
    imshow("mean", mean.reshape(1,height));
    Mat diffmat = recog::create_variance_mat(images , mean) ;
    
    if(argc == 2) {
        waitKey(0);
    }
    return 0;
}
