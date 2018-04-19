// %HISTORY%
// March 25 2018 - Pawan Negi - Code intialize
//
#include"recog.hpp"
#include<iostream>

using namespace std;
using namespace cv;
using namespace face;
namespace recog {
    //description
    //

    cv::Mat calculate_mean(std::vector<cv::Mat> images) {
        Mat mymean = images[0] * 0.0;
        for(int i = 0 ; i < images.size() ; i++){
            mymean += norm_0_255(images[i])/images.size() ;
            //cout << images.size() << endl ;
        }
        
        return mymean;
    }


    //description
    //

    cv::Mat create_variance_mat(std::vector<cv::Mat> images, cv::Mat mean) {
        
        return images[0];

    }

    //description
    //

    cv::Mat calculate_eigen(cv::Mat cov_matrix) {
        return cov_matrix;

    }


    //description
    //

    cv::Mat get_pricncipal_comp(cv::Mat eigenvec, cv::Mat eigenval) {
        return eigenvec;

    }

    //description
    //

    int recognize(cv::Mat PC, cv::Mat inputimage) {
        return 0;

    }

    //description
    //

    int inbuilt_recognition(std::vector<cv::Mat> images ,vector<int> labels , cv::Mat inputimage) {
        Ptr<BasicFaceRecognizer> model1 = FisherFaceRecognizer::create();

        Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create(); //EigenFaceRecognizer::create();
        model->train(images, labels);

        // The following line predicts the label of a given
        // test image:
        int predictedLabel = model->predict(inputimage);
        return predictedLabel;

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

    void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
        std::ifstream file(filename.c_str(), ifstream::in);
        if (!file) {
            string error_message = "No valid input file was given, please check the given filename.";
            CV_Error(CV_StsBadArg, error_message);
        }
        string line, path, classlabel;
        while (getline(file, line)) {
            stringstream liness(line);
            getline(liness, path, separator);
            getline(liness, classlabel);
            if (!path.empty() && !classlabel.empty()) {
                images.push_back(imread(path, 0));
                labels.push_back(atoi(classlabel.c_str()));
            }
        }

    }
}
