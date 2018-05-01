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

    /*description*/
    Mat detectAndDisplay(Mat frame, std::vector<Rect> *faces) {

        cv::CascadeClassifier face_cascade;
        String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
        
        //-- 1. Load the cascades
        if (!face_cascade.load(face_cascade_name)) {
            printf("--(!)Error loading\n");
            exit(-1);
        };

        Mat frame_gray;

        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        Mat cropped;
        //-- Detect faces
        face_cascade.detectMultiScale(frame_gray, *faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for (size_t i = 0; i < faces->size(); i++) {

            crop_image(&cropped, frame, (*faces)[i].x, (*faces)[i].y, (*faces)[i].width, (*faces)[i].height);
            create_rectangle((*faces)[i].x, (*faces)[i].y, (*faces)[i].width, (*faces)[i].height, frame);
        }
                //-- Show what you got
                if (cropped.empty() == 0){
                    std::ostringstream oss;
                    oss << "karhteek" << rand() % (1+100) << ".jpg" ;
                    std::string var = oss.str();
                    imwrite(var.c_str(), cropped);
                }

//        putText(frame, "Pawan", Point((*faces)[0].x, (*faces)[0].y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
//        imshow("Face detection", frame);
        cvtColor(cropped,cropped,CV_BGR2GRAY);
        //cout<<endl<<cropped.channels()<<" "<<cropped.type()<<" "<<cropped.rows<<" "<<cropped.cols<<endl;
        return cropped;
    }
}
