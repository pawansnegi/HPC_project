// %HISTORY%
// March 25 2018 - Pawan Negi - Code intialize
//
#include"detect.hpp"
#include<iostream>

using namespace std;
using namespace cv;

namespace detect {

    /*description*/
    void create_rectangle(int facex, int facey, int w, int h, Mat frame) {

        Point center1(facex, facey);
        Point center2(facex + w, facey + h);
        cv::rectangle(frame, center1, center2, Scalar(255, 0, 255), 4);

    }

    /*description*/
    Mat crop_image(Mat frame, int facex, int facey, int w, int h) {
        return frame;
    }

    /*description*/
    void detectAndDisplay(Mat frame, cv::CascadeClassifier face_cascade) {
        std::vector<Rect> faces;
        Mat frame_gray;

        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        //-- Detect faces
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (size_t i = 0; i < faces.size(); i++) {
            create_rectangle(faces[i].x, faces[i].y, faces[i].width, faces[i].height, frame);
        }
        //-- Show what you got
        imshow("Face detection", frame);
    }
}
