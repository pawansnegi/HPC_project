#include"track.hpp"
#include<iostream>

using namespace std;
using namespace cv;
namespace track{
//description
//
    void sometrackfucntion1(int test){
        Mat image;
        image = imread("3.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

        if(! image.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
	    return ;
        }

        namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        imshow( "Display window", image );                   // Show our image inside it.

        waitKey(0);                                          // Wait for a keystroke in the window
       cout << "this is sometrackfunction1" << endl ;
   }


//description
//
    void sometrackfucntion2(int test){
        sometrackfucntion1(test); 
    }
    
// description
    bool isLargeDeltaInFrames(Rect frame1, Rect frame2) {        
        int deltaw = frame1.width - frame2.width ;
        int deltah = frame1.height - frame2.height ;
        
        if (deltaw * deltaw + deltah * deltah > 25000 )
            return true ;
        else
            return false ;
    }
}
