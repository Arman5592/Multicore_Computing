
#include <stdio.h>
#include <iostream>
#include<time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define tpb 128

using namespace cv;
using namespace std;


int main()
{
  clock_t start, stop;
  string imname;
	cout << "Enter image name:";
	getline (cin, imname);

	Mat img = imread(imname,IMREAD_COLOR);
	Size s = img.size();
	int w = s.width;
	int h = s.height;

	Mat img_sobel(h, w, CV_8UC3, Scalar(0,0,0));

	unsigned char* ar_img = img.data;
	unsigned char* ar_img_inv = img_sobel.data;

	int u_char_size = h * w * 3 * sizeof(unsigned char);

	int vec_size = h * w * 3;
	int block_count = ((vec_size + tpb - 1)/tpb) + 1;

  int down = 3*w;
  int up = vec_size - 3*w;
  w = w*3;

	//Sobel<<<block_count, tpb>>>  (ar_img, ar_img_inv, vec_size, 3*w, down, up);
  start = clock();
  for(int i=0; i<vec_size; i++){
      
      if (i > down && i < up){
        int rem = i%w;
        if (rem > 2 && rem < w-3){
          //find nearby positions
          int up = i - w;
          int down = i + w;
          int left = i - 3;
          int right = i + 3;
          int upleft = i - w - 3;
          int upright = i - w + 3;
          int downleft = i + w - 3;
          int downright = i + w + 3;

          int tmp1 = -ar_img[upleft] + ar_img[upright] - 2*ar_img[left] + 2*ar_img[right] - ar_img[downleft] + ar_img[downright];
          if(tmp1<0) tmp1=0;
          if(tmp1>255) tmp1=255;

          int tmp2 = ar_img[upleft] + 2*ar_img[up] + ar_img[upright] - ar_img[downleft] - 2*ar_img[down] - ar_img[downright];
          if(tmp2<0) tmp2=0;
          if(tmp2>255) tmp2=255;

          ar_img_inv[i]=tmp1 + tmp2;

        }
        else {
          ar_img_inv[i]=ar_img[i];
        }
      }
      else {
        ar_img_inv[i]=ar_img[i];
      }
  }
  stop = clock();
  cout << "Enter output name:";
	getline (cin, imname);
  w = w/3;
	Mat output = Mat(h, w, CV_8UC3, ar_img_inv);
	imwrite(imname, output);
  cout << stop-start;

  free(ar_img);
	free(ar_img_inv);
}