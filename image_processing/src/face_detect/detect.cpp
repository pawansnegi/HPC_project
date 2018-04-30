// %HISTORY%
// March 25 2018 - Pawan Negi - Code intialize
//
#include"detect.hpp"

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
    void crop_image(Mat *out, Mat frame, int facex, int facey, int w, int h) {
        cv::Rect myROI(facex, facey, w, h);
        *out = frame(myROI);
    }

    Mat cropAndCreateRect(Mat frame , std::vector<Rect> *faces){
        Mat cropped ;
        for (size_t i = 0; i < faces->size(); i++) {

            crop_image(&cropped, frame, (*faces)[i].x, (*faces)[i].y, (*faces)[i].width, (*faces)[i].height);
            create_rectangle((*faces)[i].x, (*faces)[i].y, (*faces)[i].width, (*faces)[i].height, frame);

        }
        return cropped ;
    }
    /*description*/
    Mat detectAndDisplay(Mat frame, std::vector<Rect> *faces) {

        cv::CascadeClassifier face_cascade;

        String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

        if (!face_cascade.load(face_cascade_name)) {
            printf("--(!)Error loading\n");
            exit(-1);
        }

        Mat frame_gray;
        Mat cropped;
        Mat newframe = frame;
        bool eye = false;
        
        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        
        equalizeHist(frame_gray, frame_gray);
        
        //eye = getRotatedFrame(frame_gray, &newframe);

        detectFace(face_cascade , newframe ,
            frame_gray , faces , eye) ;

        cropped = cropAndCreateRect(frame , faces) ;
        
        //        -- Show what you got
        //        if (cropped.empty() == 0){
        //            std::ostringstream oss;
        //            oss << "karhteek" << rand() % (1+100) << ".jpg" ;
        //            std::string var = oss.str();
        //            imwrite(var.c_str(), cropped);
        //        }

        //putText(frame, "Pawan", Point((*faces)[0].x, (*faces)[0].y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
        //imshow("Face detection", frame_gray );
        //imshow("Face detection", frame);
        return cropped;
    }

    Mat getRotationMat(Mat frame) {

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

    bool getRotatedFrame(Mat frame, Mat *newframe) {

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
    
    void detectFace(cv::CascadeClassifier face_cascade , Mat rotatedframe ,
            Mat orgframe , std::vector<Rect> *faces , bool isrotated){
        
        if (isrotated) 
        {
            face_cascade.detectMultiScale(rotatedframe, *faces, 1.1);
            for (size_t i = 0; i < faces->size(); i++) 
            {
                (*faces)[i].x = 0;
                (*faces)[i].y = 0;
                cout << "eye" << endl;
            }
        } else
        {
            face_cascade.detectMultiScale(orgframe, *faces, 1.1);
        }
    }
}
