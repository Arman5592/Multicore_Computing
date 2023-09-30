
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define tpb 128

using namespace cv;

__global__ void GreenScreen(unsigned char* in, unsigned char* out, unsigned char* bg, int total_pixels) {

	int r = (blockIdx.x * blockDim.x + threadIdx.x)*3;
  int g = r+1;
  int b = r+2;

	if (b < total_pixels)
	{
		int gd = 275 + in[r] + in[b];
    int rd = 255 + in[g] + in[b];
    int bd = 255 + in[r] + in[g];

    if (gd < rd && gd < bd){
      out[r] = bg[r];
      out[g] = bg[g];
      out[b] = bg[b];
    }
    else {
      out[r] = in[r];
      out[g] = in[g];
      out[b] = in[b];
    }
	}  
}


int main()
{

	Mat img = imread("d.jpg",IMREAD_COLOR);
	Size s = img.size();
	int w = s.width;
	int h = s.height;

  Mat bg = imread("bg.jpg",IMREAD_COLOR);
	Mat img_invert(h, w, CV_8UC3, Scalar(0,0,0));

	unsigned char* char_img = img.data;
	unsigned char* new_img = img_invert.data;
  unsigned char* char_bg = bg.data;

	int u_char_size = h * w * 3 * sizeof(unsigned char);

	unsigned char *ar_img, *ar_img_inv, *ar_bg;

	int vec_size = h * w * 3;
	int block_count = ((vec_size + tpb - 1)/tpb)/3 + 1;

	cudaMalloc((void**) &ar_img, u_char_size);
	cudaMalloc((void**) &ar_img_inv, u_char_size);
  cudaMalloc((void**) &ar_bg, u_char_size);

	cudaMemcpy(ar_img, char_img, u_char_size, cudaMemcpyHostToDevice);
	cudaMemcpy(ar_img_inv, new_img, u_char_size, cudaMemcpyHostToDevice);
  cudaMemcpy(ar_bg, char_bg, u_char_size, cudaMemcpyHostToDevice);

	GreenScreen<<<block_count, tpb>>>  (ar_img, ar_img_inv, ar_bg, vec_size);

	cudaMemcpy(char_img, ar_img, u_char_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_img, ar_img_inv, u_char_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(char_bg, ar_bg, u_char_size, cudaMemcpyDeviceToHost);

	cudaFree(ar_img);
	cudaFree(ar_img_inv);
  cudaFree(ar_bg);
   
	Mat output = Mat(h, w, CV_8UC3, new_img);
	imwrite("gs.jpg", output);
}