// %HISTORY%
// March 25 2018 - Pawan Negi - Code intialize
//
#include"recog.hpp"
#include<iostream>
#include"../face_detect/detect.hpp"

using namespace std;
using namespace cv;
using namespace face;
namespace recog {
    //description
    //

    cv::Mat calculate_mean(std::vector<cv::Mat> images) {
        
        const Scalar A=0; 
        Mat mymean(112,92,CV_16UC1,A);
        mymean = mymean.reshape(1,1);
        for(int i = 0 ; i < images.size() ; i++){
                     images[i].convertTo(images[i],CV_16UC1); 
                     //images[i] = images[i].reshape(1,1);
            mymean += (images[i]);
           // cout<< mymean<<endl;
            //mymean += norm_0_255(images[i])/images.size() ;
            //cout << images.size() << endl ;
            //imshow("mymean", mymean.reshape(1, 112));
            //waitKey(0);
            
        }
        
        mymean= mymean/images.size();
        mymean=norm_0_255(mymean);
//        cout<<"im here recog"<<endl;
            // imshow("mymean", mymean.reshape(1, 112));
            //waitKey(0);
            mymean.convertTo(mymean,CV_8U);
        return  (mymean);
    }


    //description
    //

    cv::Mat create_variance_mat(std::vector<cv::Mat> images, cv::Mat mean) {
        //Mat myvariance = images[0]*0.0;
        Mat myvariance(600,600,CV_16SC3);
//        Mat matrix();
//        for(int i=0;i<images.size();i++){
//             Mat temp=norm_0_255(images[i])-mean;
//             myvariance +=(temp*temp.t)/images.size();
//
//
//        }        
        return myvariance;
        //return images[0];

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

        // The following line predicts the label of a giventest image:
        int predictedLabel = model->predict(inputimage);
        /*VideoCapture cap(0);
        Mat save_img,save_img1; cap >> save_img;
        imshow("save_img",save_img);
        waitKey(0);
        //save_img.convertTo(save_img,CV_8U);
        resize(save_img, save_img1, Size(92,112),0 , 0);
        //save_img1.convertTo(save_img1,CV_8U);
        save_img1=norm_0_255(save_img1);*/
            /*VideoCapture cap(0);
        Mat frame;

    
        imshow("frame",frame);
        waitKey(0);
        int predictedLabel = model->predict(frame);
            Mat eigenvalues = model->getEigenValues();
            Mat W = model->getEigenVectors();
            Mat mean1 = model->getMean();
            //namedWindow( "mean1", WINDOW_AUTOSIZE );
            imshow("1mean1", norm_0_255(mean1.reshape(1, images[0].rows)));
            waitKey(0);
    
    // Display or save the Eigenfaces:
    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, 112));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        // Display or save:
       
            imshow(format("eigenface_%d", i), cgrayscale);
            //waitKey(0);
    }*/

    /*// Display or save the image reconstruction at some predefined steps:
    for(int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components+=15) {
        // slice the eigenvectors from the model
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = subspaceProject(evs, mean, images[0].reshape(1,1));
        Mat reconstruction = subspaceReconstruct(evs, mean, projection);
        // Normalize the result:
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // Display or save:
        
            imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
       
        
    }*/
    // Display if we are not writing to an output folder:
    
        waitKey(0);
    
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
