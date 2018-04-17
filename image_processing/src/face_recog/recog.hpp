// %HISTORY%
// March 25 2018 - Pawan Negi - Code initilize

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

   #ifdef __cpluplus
      extern "C"
      {
   #endif

   namespace recog{
       /*function description */
       cv::Mat calculate_mean(std::vector<cv::Mat> images) ;
       /*function description */
       cv::Mat create_variance_mat(std::vector<cv::Mat> images , cv::Mat mean);
       /*function description*/
       cv::Mat calculate_eigen(cv::Mat cov_matrix) ;
       /*function description*/
       cv::Mat get_pricncipal_comp(cv::Mat eigenvec , cv::Mat eigenval);
       /*function description*/
       int recognize(cv::Mat PC , cv::Mat inputimage);

       /*inbuilt recognition*/
       int inbuilt_recognition(std::vector<cv::Mat> images ,std::vector<int> labels
       , cv::Mat inputimage);
       
       /*function*/
       cv::Mat norm_0_255(cv::InputArray _src);
       /*function*/
       void read_csv(const std::string& filename, std::vector<cv::Mat>& images, 
               std::vector<int>& labels, char separator = ';');
   }


   #ifdef __cpluplus
      } // extern "C"
   #endif

#endif // _RECOG_H
