/// @file main.cpp
/// main caller for Detection, Recognition and Tracking
///
/// @version 0.1
/// @author Pawan Negi <pawan2713@gmail.com>
///
/// &copy;2018-2019 pawannegi. All rights reserved.

/*! \mainpage Face Detection, Recognition and Tracking
 * 
 *   
 * This Projects helps to understand opencv to use it for face
 * Detection, Recognition and Tracking. The code has been written in cpp
 * language but the flow remains the same for other languages. The overall
 * flow is divided into three categories :- 
 * - \subpage "How to Detect a face in a frame?"
 *
 * - \subpage "How to Recognize a face?"
 * 
 * - \subpage "How to Track a face?"
 *
 */

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
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

//#include <OpenCL/opencl.h>
//#include <OpenCL/cl.h>

#include<mpi.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace face;

/// Recognition call for individual testing of recognition functions
/// source https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html
///
/// @return
///     void return

void recognition_call() {
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        recog::read_csv<Mat>("../data.csv", images, labels, ';');
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "../data.csv" << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    int height = images[0].rows;

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(); //EigenFaceRecognizer::create();
    model->train(images, labels);


    VideoCapture cap(0);
    Mat frame;

    //-- 2. Read the video stream
    std::vector<Rect> faces;
    std::vector<Mat> cropdframes;
    if (cap.isOpened()) {
        while (true) {
            cap >> frame;
            Mat det_face;
            faces.clear();
            cropdframes.clear() ;
            if (!frame.empty()) {
                detect::detectAndDisplay<Mat>(frame, &faces, &cropdframes);
            } else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            for (int i = 0; i < cropdframes.size(); i++) {
                Mat testSample = cropdframes[i];
                if (!testSample.empty()) {
                    resize(testSample, testSample, Size(112, 92), 0, 0);
                    testSample.convertTo(testSample, CV_8U);

                    int predictedLabel = -123 ; //model->predict(testSample);
                    double confidence = 0 ;
                    model->predict(testSample , predictedLabel , confidence) ;

                    //cout << model->getThreshold() << endl ;
                    //cout << model->getNumComponents() << endl ;
                    string result_message = format("Predicted class = %d with "
                            "confidence = %lf", predictedLabel , confidence);
                    cout << result_message << endl;

                    if (predictedLabel == 0 && confidence < 100) {
                        putText(frame, "Kartheek", Point(faces[i].x, faces[i].y)
                                , FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
                    }
                    else if (predictedLabel == 1 && confidence < 100) {
                        putText(frame, "Pawan", Point(faces[i].x, faces[i].y)
                                , FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
                    }
                    else{
                        putText(frame, "Unknown", Point(faces[i].x, faces[i].y)
                                , FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
                    }
                }
            }

            imshow("Face detection", frame);

            int c = waitKey(10);
            if ((char) c == 'c') {
                break;
            }

        }
    }

    //    for (int i = 0; i < images.size(); i++) {
    //        images[i] = images[i].reshape(1, 1);
    //    }
    //
    //    Mat mean = recog::calculate_mean(images);
    //    //cout << mean.at<double>(112) << endl;
    //    imshow("2mean", mean.reshape(1, height));
    //    Mat diffmat = recog::create_variance_mat(images, mean);
    //
    //    Mat x;
    //    waitKey(0);
    //return predictedLabel;

}

void recognition_call_ocl() {
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        recog::read_csv<Mat>("../data.csv", images, labels, ';');
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "../data.csv" << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    int height = images[0].rows;

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(); //EigenFaceRecognizer::create();
    model->train(images, labels);


    VideoCapture cap(0);
    UMat frame;

    //-- 2. Read the video stream
    std::vector<Rect> faces;
    std::vector<UMat> cropdframes;
    if (cap.isOpened()) {
        while (true) {
            cap >> frame;
            UMat det_face;
            faces.clear();
            cropdframes.clear() ;
            if (!frame.empty()) {
                detect::detectAndDisplay<UMat>(frame, &faces, &cropdframes);
            } else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            for (int i = 0; i < cropdframes.size(); i++) {
                UMat testSample = cropdframes[i];
                if (!testSample.empty()) {
                    resize(testSample, testSample, Size(112, 92), 0, 0);
                    testSample.convertTo(testSample, CV_8U);

                    int predictedLabel = -123 ; //model->predict(testSample);
                    double confidence = 0 ;
                    model->predict(testSample , predictedLabel , confidence) ;

                    //cout << model->getThreshold() << endl ;
                    //cout << model->getNumComponents() << endl ;
                    string result_message = format("Predicted class = %d with "
                            "confidence = %lf", predictedLabel , confidence);
                    cout << result_message << endl;

                    if (predictedLabel == 0 && confidence < 100) {
                        putText(frame, "Kartheek", Point(faces[i].x, faces[i].y)
                                , FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
                    }
                    else if (predictedLabel == 1 && confidence < 100) {
                        putText(frame, "Pawan", Point(faces[i].x, faces[i].y)
                                , FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
                    }
                    else{
                        putText(frame, "Unknown", Point(faces[i].x, faces[i].y)
                                , FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
                    }
                }
            }

            imshow("Face detection", frame);

            int c = waitKey(10);
            if ((char) c == 'c') {
                break;
            }

        }
    }

    //    for (int i = 0; i < images.size(); i++) {
    //        images[i] = images[i].reshape(1, 1);
    //    }
    //
    //    Mat mean = recog::calculate_mean(images);
    //    //cout << mean.at<double>(112) << endl;
    //    imshow("2mean", mean.reshape(1, height));
    //    Mat diffmat = recog::create_variance_mat(images, mean);
    //
    //    Mat x;
    //    waitKey(0);
    //return predictedLabel;

}

/// Detection call for individual testing of face detection functions
/// source https://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
///
/// @return
///     void return

void detection_call() {

    VideoCapture cap(0);
    Mat frame;

    //-- 2. Read the video stream
    std::vector<Rect> faces;
    if (cap.isOpened()) {
        while (true) {
            cap >> frame;
            Mat x;
            //-- 3. Apply the classifier to the frame
            if (!frame.empty()) {
                detect::detectAndDisplay<Mat>(frame, &faces);
            } else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            imshow("Face detection", frame);

            int c = waitKey(1);
            if ((char) c == 'c') {
                break;
            }

        }
    }
}

/// Detection call for individual testing of face detection functions for opencl
/// 
/// @return
///     void return

void detection_call_ocl() {
    VideoCapture cap(0);
    UMat frame;

    Mat frame2;

    std::vector<Rect> faces;
    if (cap.isOpened()) {
        while (true) {
            cap.read(frame);

            if (!frame.empty()) {
                detect::detectAndDisplay<UMat>(frame, &faces);
            } else {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            frame.copyTo(frame2);
            imshow("Face detection", frame2);

            int c = waitKey(1);
            if ((char) c == 'c') {
                break;
            }
            imshow("Face detection", frame);
        }
    }
}

/// tracking call for individual testing of tracking functions
/// source https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
/// @return
///     void return

void tracking_call() {

    // Create a tracker
    Ptr<Tracker> tracker;
    tracker = TrackerMedianFlow::create();
    // Read video

    VideoCapture video(0);
    Mat frame;
    std::vector<Rect> faces;
    bool detect = true;
    bool trackfail = false;
    int wait = 0;
    int detectagain = 0;

    Rect2d bbox(287, 23, 86, 320);
    // Exit if video is not opened
    if (video.isOpened()) {
        video >> frame;
        detect::detectAndDisplay<Mat>(frame, &faces);

        while (true) {
            //            detectagain++;
            video >> frame;
            faces.erase(faces.begin(), faces.end());
            //
            cout << track::isLargeDeltaInFrames(bbox, faces[0]) << endl;
            //            if ((detect && !track::isLargeDeltaInFrames(bbox , faces[0])) || 
            //                    ( wait < 5)) {
            //                tracker->clear();
            //                bbox.x = faces[0].x;
            //                bbox.y = faces[0].y;
            //                bbox.height = faces[0].height;
            //                bbox.width = faces[0].width;
            //
            //                tracker = TrackerMedianFlow::create();
            //                tracker->init(frame, bbox);
            //                detect = false;
            //                wait++ ;
            //            }else{
            //                trackfail = true ;
            //            }
            ////            else{
            ////                tracker->clear();
            ////                tracker = TrackerMedianFlow::create();
            ////                tracker->init(frame, bbox);
            ////                tracker->update(frame, bbox);
            ////            }
            //            
            //            tracker->update(frame, bbox);
            //            cout << "updating = " << bbox.x << " " << bbox.y << " "
            //                    << " " << detect << " " << bbox.width << endl;
            //            if (trackfail || track::isBoxAtCorner(bbox , frame.rows , frame.cols)) {
            detect::detectAndDisplay<Mat>(frame, &faces);

            stringstream ss;
            ss << faces[0].x << " " << faces[0].width;
            string abc = ss.str();

            putText(frame, abc.c_str(), Point(faces[0].x + 50, faces[0].y + 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
            //                detectagain = 0;
            //                detect = true;
            //                trackfail = false ;
            //            } else {
            //                putText(frame, "tracked", Point(bbox.x + 50, bbox.y + 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(256, 0, 0), 2);
            //                rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
            //                //printf(" --(!) No captured frame -- Break!");
            //                //break;
            //            }
            //

            putText(frame, "Pawan", Point((faces)[0].x, (faces)[0].y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
            cout << faces[0].x << endl;
            //            imshow("Face detection", frame);
            int c = waitKey(10);
            if ((char) c == 'c') {
                break;
            }
        }
    } else {
        cout << "can't open vedio" << endl;
    }
}

/// main function
/// @return
///     exit number

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
    recognition_call_ocl();
    //detection_call_ocl() ;
    //detection_call();
    //tracking_call();

    //    cv::ocl::Context context;
    //    std::vector<cv::ocl::PlatformInfo> platforms;
    //    cv::ocl::getPlatfomsInfo(platforms);
    //    for (size_t i = 0; i < platforms.size(); i++)
    //    {
    //        //Access to Platform
    //        const cv::ocl::PlatformInfo* platform = &platforms[i];
    //
    //        //Platform Name
    //        std::cout << "Platform Name: " << platform->name().c_str() << "\n" << endl;
    //
    //        //Access Device within Platform
    //        cv::ocl::Device current_device;
    //        for (int j = 0; j < platform->deviceNumber(); j++)
    //        {
    //            //Access Device
    //            platform->getDevice(current_device, j);
    //            int deviceType = current_device.type();
    //            cout << "devide type " << context.ndevices() << endl ;
    //            cout << "Device name:  " << current_device.name() << endl;
    //            if (deviceType == 2)
    //                cout << context.ndevices() << " CPU devices are detected." << std::endl;
    //            if (deviceType == 4)
    //                cout << context.ndevices() << " GPU devices are detected." << std::endl;
    //            cout << "===============================================" << endl << endl;
    //            
    //            //cin.ignore(1);
    //        }
    //    }

    return 0;
}