#include"track.hpp"
#include<iostream>

using namespace std;
using namespace cv;
namespace track{
// description
    bool isLargeDeltaInFrames(Rect frame1, Rect frame2) {        
        int deltaw = frame1.width - frame2.width ;
        int deltah = frame1.height - frame2.height ;
        
        if (deltaw * deltaw + deltah * deltah > 25000 )
            return true ;
        else
            return false ;
    }
    
    bool isBoxAtCorner(Rect frame1 , int rows , int cols){
        
        int x1 = frame1.x ;
        int y1 = frame1.y ;
        int x2 = frame1.x + frame1.width ;
        int y2 = frame1.y + frame1.height ;
        
        if(x1 < 1 || x2 > rows -1  || y1 <  1 || y2 > cols - 1  )
            return true ;
        else
            return false ;
        
    }
}
