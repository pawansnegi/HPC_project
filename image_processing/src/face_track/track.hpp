// %HISTORY%
// March 25 2018 - Pawan Negi - Code initilize

#ifndef _TRACK_H
#define _TRACK_H

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

#include<mpi.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
   #ifdef __cpluplus
      extern "C"
      {
   #endif

   namespace track{
       // function description
       void sometrackfucntion1(int test) ;
       // function description
       void sometrackfucntion2(int test) ;
       //fucntion description
       bool isLargeDeltaInFrames(Rect frame1, Rect frame2);


   }


   #ifdef __cpluplus
      } // extern "C"
   #endif

#endif // _TRACK_H
