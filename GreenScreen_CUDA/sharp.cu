
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define tpb 128

using namespace cv;

__global__ void Sharpen(unsigned char* in, unsigned char* out, int total_pixels, int w, int down, int up) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i > down && i < up){
    if (i%w !=0 && i%w != w-1){
      //find nearby positions
      int up = i - 3*w;
      int down = i + 3*w;
      int left = i - 3;
      int right = i + 3;

      int tmp = 5*in[i]-in[up]-in[down]-in[left]-in[right];
      if(tmp<0) tmp=0;
      if(tmp>255) tmp=255;

      out[i]=tmp;

      //out[i]=in[i];
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

	Mat img = imread("gs.jpg",IMREAD_COLOR);
	Size s = img.size();
	int w = s.width;
	int h = s.height;

	Mat img_invert(h, w, CV_8UC3, Scalar(0,0,0));

	unsigned char* char_img = img.data;
	unsigned char* new_img = img_invert.data;

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

	Sharpen<<<block_count, tpb>>>  (ar_img, ar_img_inv, vec_size, w, down, up);

	cudaMemcpy(char_img, ar_img, u_char_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(new_img, ar_img_inv, u_char_size, cudaMemcpyDeviceToHost);

	cudaFree(ar_img);
	cudaFree(ar_img_inv);
   
	Mat output = Mat(h, w, CV_8UC3, new_img);
	imwrite("sharp.jpg", output);
}