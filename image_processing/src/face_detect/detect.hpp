/// @file detect.hpp
/// face detection functions
///
/// @version 0.1
/// @author Pawan Negi <pawan2713@gmail.com>
///
/// &copy;2018-2019 pawannegi. All rights reserved.

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
using namespace std;
#ifdef __cpluplus
extern "C" {
#endif

    namespace detect {
        
/// Detects face using a CascadeClassifier. Two frame are given as an input
/// depending upon whether the face is rotated or not the CascadeClassifier is
/// on the respective frame.
///
/// @param[in] face_cascade
///     CascadeClassifier which can detect a face in a frame.
/// @param[in] rotatedframe
///     rotated frame if there is some rotation in the frame
/// @param[in] orgframe
///     original frame without the rotation.
/// @param[out] faces
///     all the detected faces bounding boxes.       
/// @param[in] isrotated
///     is rotation is done or not.
///
/// @return
///     void

        template <class mattype>
        void detectFace(cv::CascadeClassifier face_cascade, mattype rotatedframe,
                mattype orgframe, std::vector<Rect> *faces, bool isrotated) {

            if (isrotated) {
                face_cascade.detectMultiScale(rotatedframe, *faces, 1.1);
                for (size_t i = 0; i < faces->size(); i++) {
                    (*faces)[i].x = 0;
                    (*faces)[i].y = 0;
                }
            } else {
                face_cascade.detectMultiScale(orgframe, *faces, 1.1);
            }
        }

/// Creates rectangles using the face bounding boxes properties
///
/// @param[in] facex
///     x coordinate of the box
/// @param[in] facey
///     y coordinate of the box
/// @param[in] w
///     width of box
/// @param[out] h
///     height of box       
/// @param[in] frame
///     frame on which rectangle to be made
///
/// @return
///     void
        template <class mattype>
        void create_rectangle(int facex, int facey, int w, int h, mattype frame) {

            Point center1(facex, facey);
            Point center2(facex + w, facey + h);
            cv::rectangle(frame, center1, center2, Scalar(255, 0, 255), 4);

        }

/// crops the frame using the bounding box properties
///
/// @param[out] out
///     cropped output frame
/// @param[in] frame
///     frame to be cropped
/// @param[in] facex
///     x coordinate of the box
/// @param[in] facey
///     y coordinate of the box
/// @param[in] w
///     width of box
/// @param[out] h
///     height of box 
///
/// @return
///     void
        template <class mattype>
        void crop_image(mattype *out, mattype frame, int facex, int facey, int w, int h) {
            cv::Rect myROI(facex, facey, w, h);
            *out = frame(myROI);
        }

/// function calling crop_image and create_rectangle
///
/// @param[in] frame
///     original image frame
/// @param[in] faces
///     faces bounding  box vectors
///
/// @return
///     cropped image
        
        template <class mattype>
        void cropAndCreateRect(mattype frame, std::vector<Rect> *faces 
                  , std::vector<mattype> *cropped) {
            mattype crop;
            for (size_t i = 0; i < faces->size(); i++) {
                
                crop_image(&crop, frame, (*faces)[i].x, (*faces)[i].y, (*faces)[i].width, (*faces)[i].height);
                create_rectangle((*faces)[i].x, (*faces)[i].y, (*faces)[i].width, (*faces)[i].height, frame);
                cropped->push_back(crop) ;
            }
        }

/// Detects face using a CascadeClassifier. calls of eye detection and make possible rotation
/// and detect the face using detectFace
///
/// @param[in] frame
///     original image frame
/// @param[out] faces
///     faces bounding  box vectors
///
/// @return
///     cropped image
        
        template <class mattype>
        void detectAndDisplay(mattype frame, std::vector<Rect> *faces 
             , std::vector<mattype> *cropped = NULL) {

            cv::CascadeClassifier face_cascade;

            String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

            if (!face_cascade.load(face_cascade_name)) {
                printf("--(!)Error loading\n");
                exit(-1);
            }

            mattype frame_gray;
            //mattype cropped;
            mattype newframe = frame;
            bool eye = false;

            cvtColor(frame, frame_gray, CV_BGR2GRAY);

            equalizeHist(frame_gray, frame_gray);

            //eye = getRotatedFrame(frame_gray, &newframe);

            detectFace(face_cascade, newframe,
                    frame_gray, faces, eye);

            cropAndCreateRect(frame, faces , cropped);
            for (int i = 0 ; i < cropped->size() ; i++){
                mattype crop = (*cropped)[i];
                if (!crop.empty()){
                    cvtColor(crop, crop, CV_BGR2GRAY);
                    (*cropped)[i] = crop ;
                }
            }

        }

/// Detect the eye in the frame and create a rotation matrix for the frame
///
/// @param[in] frame
///     original frame on which face and eyes are visible
///
/// @return
///     rotation matrix
        
        template <class mattype>
        Mat getRotationMat(mattype frame) {

            Mat tranmat;
            CascadeClassifier eyes_cascade;
            String eyes_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

            if (!eyes_cascade.load(eyes_cascade_name)) {
                printf("--(!)Error loading\n");
                exit(-1);
            }

            Point2f center;
            std::vector<Rect> eyes;
            eyes_cascade.detectMultiScale(frame, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

            if (eyes.size() > 0) {
                double x1 = eyes[0].x + eyes[0].width / 2;
                double y1 = eyes[0].y + eyes[0].height / 2;
                double x2 = 0, y2 = 0;
                for (int i = 1; i < eyes.size(); i++) {
                    x2 = eyes[i].x + eyes[i].width / 2;
                    y2 = eyes[i].y + eyes[i].height / 2;
                    double dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                    if (dist > 200) break;
                }
                if (x2 != 0 && y2 != 0) {
                    double angle = atan((y2 - y1) / (x2 - x1));
                    center.x = (x1 + x2) / 2;
                    center.y = (y1 + y2) / 2;
                    cout << x1 << " " << y1 << endl;
                    cout << x2 << " " << y2 << endl;
                    tranmat = cv::getRotationMatrix2D(center, angle * 180 / 3.14, 1.0);
                }
            }
            return tranmat;
        }

/// Takes the rotation matrix and rotates the frame if necessary
/// if rotation is nor required returns false.
///
/// @param[in] frame
///     original frame on which face and eyes are visible
/// @param[out] newframe
///     rotated frame out
///
/// @return
///     true/false
        
        template <class mattype>
        bool getRotatedFrame(mattype frame, mattype *newframe) {

            Mat tranmat = getRotationMat(frame);

            if (!tranmat.empty()) {
                double sinv, cosv;
                int width = frame.size().width, height = frame.size().height;
                sinv = tranmat.at<double>(0, 1);
                cosv = tranmat.at<double>(0, 0);

                Size dstSize(width * cosv + height*sinv, width * sinv + height * cosv);

                cout << dstSize << endl;
                tranmat.at<double>(0, 2) += (width * cosv + height * sinv) / 2 - width / 2;
                tranmat.at<double>(1, 2) += (width * sinv + height * cosv) / 2 - height / 2;

                cv::warpAffine(frame, *newframe, tranmat, Size(abs(dstSize.height), abs(dstSize.width)));

                return true;
            } else
                return false;
        }
    }


#ifdef __cpluplus
} // extern "C"
#endif

#endif // _DETECT_H
