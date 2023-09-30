
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define tpb 128

using namespace cv;
using namespace std;

__global__ void Sobel(unsigned char* in, unsigned char* out, int total_pixels, int w, int down, int up) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
  
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

      int tmp1 = -in[upleft] + in[upright] - 2*in[left] + 2*in[right] - in[downleft] + in[downright];
      if(tmp1<0) tmp1=0;
      if(tmp1>255) tmp1=255;

      int tmp2 = in[upleft] + 2*in[up] + in[upright] - in[downleft] - 2*in[down] - in[downright];
      if(tmp2<0) tmp2=0;
      if(tmp2>255) tmp2=255;

      out[i]=tmp1 + tmp2;

    }
    else {
      out[i]=in[i];
    }
  }
  else {
    out[i]=in[i];
  }
  
}


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

	unsigned char* char_img = img.data;
	unsigned char* new_img = img_sobel.data;

	int u_char_size = h * w * 3 * sizeof(unsigned char);

	unsigned char *ar_img, *ar_img_inv;

	int vec_size = h * w * 3;
	int block_count = ((vec_size + tpb - 1)/tpb) + 1;

	cudaMalloc((void**) &ar_img, u_char_size);
	cudaMalloc((void**) &ar_img_inv, u_char_size);

	cudaMemcpy(ar_img, char_img, u_char_size, cudaMemcpyHostToDevice);
	cudaMemcpy(ar_img_inv, new_img, u_char_size, cudaMemcpyHostToDevice);

  int down = 3*w;
  int up = vec_size - 3*w;

  start = clock();
	Sobel<<<block_count, tpb>>>  (ar_img, ar_img_inv, vec_size, 3*w, down, up);
  stop = clock();

	cudaMemcpy(char_img, ar_img, u_char_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_img, ar_img_inv, u_char_size, cudaMemcpyDeviceToHost);

	cudaFree(ar_img);
	cudaFree(ar_img_inv);
  
  cout << "Enter output name:";
	getline (cin, imname);
	Mat output = Mat(h, w, CV_8UC3, new_img);
	imwrite(imname, output);
  cout << stop - start;
}