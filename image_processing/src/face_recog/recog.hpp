/// @file recog.hpp
/// face recognition functions
///
/// @version 0.1
/// @author Pawan Negi <pawan2713@gmail.com>
///
/// &copy;2018-2019 pawannegi. All rights reserved.

#ifndef _RECOG_H
#define _RECOG_H

#include <opencv2/core/core.hpp>
#include <opencv2/face.hpp>
#include "opencv2/face/facerec.hpp"
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include<mpi.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include<sys/stat.h>

using namespace std;
using namespace cv;
using namespace face;

   #ifdef __cpluplus
      extern "C"
      {
   #endif

   namespace recog{

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
       mattype calculate_mean(std::vector<mattype> images) {
        
        const Scalar A=0; 
        mattype mymean(112,92,CV_16UC1,A);
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
    mattype create_variance_mat(std::vector<mattype> images, mattype mean) {
        //Mat myvariance = images[0]*0.0;
        mattype myvariance(600,600,CV_16SC3);
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
    mattype calculate_eigen(mattype cov_matrix) {

        return cov_matrix;

    }


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
    mattype get_pricncipal_comp(mattype eigenvec, mattype eigenval) {
        return eigenvec;

    }

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
    int recognize(mattype PC, mattype inputimage) {
        return 0;

    }

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
    int inbuilt_recognition(std::vector<mattype> images ,vector<int> labels , mattype inputimage) {
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
    
       // waitKey(0);
    
        return predictedLabel;

    }
    
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
    mattype norm_0_255(InputArray _src) {
        mattype src = _src.getMat();
        // Create and return normalized image:
        mattype dst;
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
    void read_csv(const string& filename, vector<mattype>& images, vector<int>& labels, char separator ) {
        std::ifstream file(filename.c_str(), ifstream::in);
        if (!file) {
            string error_message = "No valid input file was given, please check the given filename.";
            CV_Error(CV_StsBadArg, error_message);
        }
        string line, path, classlabel;
        struct stat buf;
        while (getline(file, line)) {
            stringstream liness(line);
            getline(liness, path, separator);
            getline(liness, classlabel);
            if (!path.empty() && !classlabel.empty()) {
                
                if (stat(path.c_str(), &buf) == 0) {
                        images.push_back(imread(path, 0));
                        labels.push_back(atoi(classlabel.c_str()));
                }else{
                    cout << "cannot read file in data.csv " << path << endl ; 
                    exit(-1);
                }
            }
        }

    }
   }


   #ifdef __cpluplus
      } // extern "C"
   #endif

#endif // _RECOG_H
