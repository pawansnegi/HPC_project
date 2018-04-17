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
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include<mpi.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace face;

void recognition_call() {
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        recog::read_csv("../data.csv", images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "../data.csv" << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    int height = images[0].rows;
    Mat testSample = images[11];
    int testLabel = labels[11];
    images.pop_back();
    labels.pop_back();

    int predictedLabel = recog::inbuilt_recognition(images, labels, testSample);
    string result_message = format("Predicted class = %d / Actual class = %d."
            , predictedLabel, testLabel);
    cout << result_message << endl;


    for (int i = 0; i < images.size(); i++) {
        images[i] = images[i].reshape(1, 1);
    }

    Mat mean = recog::calculate_mean(images);
    cout << mean.at<double>(112) << endl;
    imshow("mean", mean.reshape(1, height));
    Mat diffmat = recog::create_variance_mat(images, mean);

    waitKey(0);

}

void detection_call() {
    VideoCapture cap(0);
    Mat frame;

    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    /** Global variables */
    String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading\n");
        exit(-1);
    };

    //-- 2. Read the video stream

    if (cap.isOpened()) {
        while (true) {
            cap >> frame;

            //-- 3. Apply the classifier to the frame
            if (!frame.empty()) {
                detect::detectAndDisplay(frame, face_cascade);
            } else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            int c = waitKey(10);
            if ((char) c == 'c') {
                break;
            }
        }
    }
}

void tracking_call() {

    cout << "tracking call" << endl;
}

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

    //recognition_call();
    detection_call();
    //tracking_call();

    return 0;
}
