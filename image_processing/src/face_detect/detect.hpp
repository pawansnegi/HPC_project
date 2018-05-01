// %HISTORY%
// March 25 2018 - Pawan Negi - Code initilize

#ifndef _DETECT_H
#define _DETECT_H

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
#ifdef __cpluplus
extern "C" {
#endif

    namespace detect {

        /*function description */
        cv::Mat detectAndDisplay(cv::Mat frame, std::vector<cv::Rect> *faces);
        /*function*/
        void create_rectangle(int facex , int facey , int w , int h , cv::Mat frame) ;
        
        Mat cropAndCreateRect(Mat frame , std::vector<Rect> *faces);
        /*function*/
        void crop_image(cv::Mat *out , cv::Mat frame , int facex , int facey , int w , int h);

        Mat getRotationMat(Mat frame );
        
        bool getRotatedFrame(Mat frame , Mat *newframe);
        
        void detectFace(cv::CascadeClassifier face_cascade , Mat rotatedframe ,
            Mat orgframe , std::vector<Rect> *faces , bool isrotated) ;
    }


#ifdef __cpluplus
} // extern "C"
#endif

#endif // _DETECT_H
