
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define tpb 128

using namespace cv;
using namespace std;

__global__ void Threshold(unsigned char* in, unsigned char* out, int total_pixels, int b) {

	int position = blockIdx.x * blockDim.x + threadIdx.x;

	if (position < total_pixels)
	{
    if (in[position] <= b)
      out[position]=0;
    else
      out[position]=255;
	}  
}


int main()
{

  string imname;
	cout << "Enter image name:";
	getline (cin, imname);

  int b;
  char trash;
	cout << "Enter threshold boundary:";
	scanf("%d", &b);
  scanf("%c", &trash);

  if(b<0) b=0;
  if(b>255) b=255;

	Mat img = imread(imname,IMREAD_COLOR);
	Size s = img.size();
	int w = s.width;
	int h = s.height;
	Mat img_invert(h, w, CV_8UC3, Scalar(0,0,0));

	unsigned char* char_img = img.data;
	unsigned char* new_img = img_invert.data;

	int u_char_size = h * w * 3 * sizeof(unsigned char);

	unsigned char *ar_img, *ar_img_inv;

	int vec_size = h * w * 3;
	int block_count = (vec_size + tpb - 1)/tpb;

	cudaMalloc((void**) &ar_img, u_char_size);
	cudaMalloc((void**) &ar_img_inv, u_char_size);

	cudaMemcpy(ar_img, char_img, u_char_size, cudaMemcpyHostToDevice);
	cudaMemcpy(ar_img_inv, new_img, u_char_size, cudaMemcpyHostToDevice);

	Threshold<<<block_count, tpb>>>  (ar_img, ar_img_inv, vec_size, b);

	cudaMemcpy(char_img, ar_img, u_char_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_img, ar_img_inv, u_char_size, cudaMemcpyDeviceToHost);

	cudaFree(ar_img);
	cudaFree(ar_img_inv);
   
	cout << "Enter output name:";
	getline (cin, imname);
	Mat output = Mat(h, w, CV_8UC3, new_img);
	imwrite(imname, output);
}