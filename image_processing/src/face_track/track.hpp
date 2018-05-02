/// @file track.hpp
/// face tracking functions
///
/// @version 0.1
/// @author Pawan Negi <pawan2713@gmail.com>
///
/// &copy;2018-2019 pawannegi. All rights reserved.

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
/// for the two rectangles calculate the delta in dimensions
///
/// @param[in] frame1
///     bounding box for first frame
/// @param[in] frame2
///     bounding box for second frame
///
/// @return
///     true/false
       bool isLargeDeltaInFrames(Rect frame1, Rect frame2);

/// Detects whether the box is at corner of the frame
///
/// @return
///     void
       bool isBoxAtCorner(Rect frame1 , int rows , int cols);

   }


   #ifdef __cpluplus
      } // extern "C"
   #endif

#endif // _TRACK_H
