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

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

void recognition_call() {//int recognition_call(cv::Mat frame){//void recognition_call() {

    
        /*originial source
     https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html
     */
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
    cout<<images.size()<<endl;
    // Quit if there are not enough images for this demo.
    if (images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    int height = images[0].rows;
    //imshow("fum",images[0]);waitKey(0);
    //cout<<endl <<images[0].type()<<" "<<images[0].channels()<<" "<<images[0].rows<<"   "<<images[0].cols<<endl;
        Ptr<BasicFaceRecognizer> model = FisherFaceRecognizer::create(); //EigenFaceRecognizer::create();
        model->train(images, labels);


    VideoCapture cap(0);
    Mat frame;

    //-- 2. Read the video stream
    std::vector<Rect> faces;
    if (cap.isOpened()) {
        while (true) {
            cap >> frame;
            Mat det_face;
            faces.empty();
            cout<<faces.size()<<endl;
            //-- 3. Apply the classifier to the frame
            //if (!frame.empty()) {
                det_face= detect::detectAndDisplay(frame, &faces);
            //} else {
            //    printf(" --(!) No captured frame -- Break!");
            //    break;
            //}
            //    putText(frame, "kartheek", Point(0,0), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
            //cout << faces[0].x << endl;
//            imshow("Face detection", frame);
            //waitKey(10);
            Mat testSample = det_face;
//            images.pop_back();
//            labels.pop_back();
           if(!testSample.empty()){ 
            resize(testSample, testSample, Size(112,92),0 , 0);
            //imshow("fum1",testSample);waitKey(0);
            testSample.convertTo(testSample,CV_8U);

    //cout<<endl<<testSample.type()<<" "<<testSample.channels()<<" " <<testSample.rows<<"   "<<testSample.cols<<endl;
    //cout<<endl <<testSample.rows<<"   "<<testSample.cols<<endl;

        int predictedLabel = model->predict(testSample);//recog::inbuilt_recognition(images, labels, testSample);
        
        string result_message = format("Predicted class = %d ",predictedLabel);/// Actual class = %d.", predictedLabel, testLabel);
         cout << result_message << endl;
           
        if(predictedLabel==0){
                putText(frame, "kartheek", Point(100,100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
            }
        if(predictedLabel==1){
                putText(frame, "pawan", Point(100,100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
            }
           }
             imshow("Face detection", frame);
           
//            if(pL==1){
//                putText(frame, "kartheek", Point(0,0), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
//            }
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

    Mat norm_0_255(InputArray _src) {
        Mat src = _src.getMat();
        // Create and return normalized image:
        Mat dst;
        switch (src.channels()) {
            case 1:
                cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
                break;
            case 3:
                cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
                break;
            default:
                src.copyTo(dst);
                break;
        }
        return dst;
    }

void detection_call() {//cv::Mat detection_call(){//void detection_call() {
    /*original source
     https://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
     */
    VideoCapture cap(0);
    Mat frame;

    //-- 2. Read the video stream
    std::vector<Rect> faces;
    if (cap.isOpened()) {
        while (true) {
            cap >> frame;
            Mat x;
            //-- 3. Apply the classifier to the frame
            //if (!frame.empty()) {
                x= detect::detectAndDisplay(frame, &faces);
            //} else {
            //    printf(" --(!) No captured frame -- Break!");
            //    break;
            //}
            //    putText(frame, "kartheek", Point(0,0), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

            //cout << faces[0].x << endl;
            imshow("Face detection", frame);
            //waitKey(10);
            
            //int pL;pL=recognition_call(x);
//            if(pL==1){
//                putText(frame, "kartheek", Point(0,0), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
//            }
            int c = waitKey(1);
            if ((char) c == 'c') {
                break;
            }
            
        }
    }
    //return frame;
}

void tracking_call() {

    /*original source 
     https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
     */
    // Create a tracker
    Ptr<Tracker> tracker;
    tracker = TrackerMedianFlow::create();
    // Read video
    
    VideoCapture video(0);
    Mat frame;
    std::vector<Rect> faces;
    bool detect = true;
    bool trackfail = false ;
    int wait = 0 ;
    int detectagain = 0;

    Rect2d bbox(287, 23, 86, 320);
    // Exit if video is not opened
    if (video.isOpened()) {
        video >> frame;
        detect::detectAndDisplay(frame, &faces);
        
        while (true) {
//            detectagain++;
            video >> frame;
            faces.erase(faces.begin() , faces.end());
//
//            cout << track::isLargeDeltaInFrames(bbox , faces[0]) << endl ;
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
                Mat cropped = detect::detectAndDisplay(frame, &faces);
                
                stringstream ss ;
                ss << faces[0].x << " " << faces[0].width ;
                string abc = ss.str() ;
                
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
//
//            putText(frame, "Pawan", Point((faces)[0].x, (faces)[0].y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
//            cout << faces[0].x << endl;
//<<<<<<< HEAD
            
//=======
//           imshow("Face detection", frame);
//>>>>>>> d3126d1f99eeff41491dbc33775b8f6533648612
            int c = waitKey(10);
            if ((char) c == 'c') {
                break;
            }
            imshow("Face detection", frame);
        }
    } else {
        cout << "can't open video" << endl;
    }
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
    //Mat x=detection_call();
    recognition_call();
    //tracking_call();
    //detection_call();
    //tracking_call();
    
    return 0;
}