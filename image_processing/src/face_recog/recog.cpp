// %HISTORY%
// March 25 2018 - Pawan Negi - Code intialize
//
#include"recog.hpp"
#include<iostream>

using namespace std;
using namespace cv;
namespace recog{
//description
//
    void somerecogfucntion1(int test) {
        Mat image;
        image = imread("5.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

        if(! image.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
	    return ;
        }

        namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
        imshow( "Display window", image );                   // Show our image inside it.

        waitKey(0);                                          // Wait for a keystroke in the window
        cout << "this is somerecogfunction1" << endl ;
    }


//description
//
    void somerecogfucntion2(int test) {
        somerecogfucntion1(test); 
}
}
