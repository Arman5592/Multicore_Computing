#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmp.c"
#include <assert.h> 

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;

__m256i v_is_green(__m256i r, __m256i g, __m256i b){
  __m256i green_dist = _mm256_add_epi32(_mm256_add_epi32(r, b), _mm256_set1_epi32(275));
  __m256i red_dist = _mm256_add_epi32(_mm256_add_epi32(g, b), _mm256_set1_epi32(255));
  __m256i blue_dist = _mm256_add_epi32(_mm256_add_epi32(r, g), _mm256_set1_epi32(255));

  __m256i cond1 = _mm256_cmpgt_epi32(red_dist, green_dist);
  __m256i cond2 = _mm256_cmpgt_epi32(blue_dist, green_dist);

  return _mm256_and_si256(cond1, cond2);
}

int main()
{
  /* start reading the file and its information*/
  byte *pixels_top, *pixels_bg;
  int32 width_top, width_bg;
  int32 height_top, height_bg;
  int32 bytesPerPixel_top, bytesPerPixel_bg;
  ReadImage("dino.bmp", &pixels_top, &width_top, &height_top, &bytesPerPixel_top);
  ReadImage("parking.bmp", &pixels_bg, &width_bg, &height_bg, &bytesPerPixel_bg);

  /* images should have color and be of the same size */
  assert(bytesPerPixel_top == 3);
  assert(width_top == width_bg);
  assert(height_top == height_bg); 
  assert(bytesPerPixel_top == bytesPerPixel_bg); 

  /* we can now work with one size */
  int32 width = width_top, height = height_top, bpp = bytesPerPixel_top; 
  
  for (int i = 0; i < height; i+=8){
    for (int j = 0; j < width; j ++){
      int start = i * width + j;
      int start1 = start + width;
      int start2 = start1 + width;
      int start3 = start2 + width;
      int start4 = start3 + width;
      int start5 = start4 + width;
      int start6 = start5 + width;
      int start7 = start6 + width;      
        
      __m256i v_top_r = _mm256_setr_epi32(pixels_top[start * bpp], pixels_top[(start1) * bpp],
                                      pixels_top[(start2) * bpp], pixels_top[(start3) * bpp],
                                      pixels_top[(start4) * bpp], pixels_top[(start5) * bpp],
                                      pixels_top[(start6) * bpp], pixels_top[(start7) * bpp]);
      
      __m256i v_top_g = _mm256_setr_epi32(pixels_top[start * bpp+1], pixels_top[(start1) * bpp+1],
                                      pixels_top[(start2) * bpp+1], pixels_top[(start3) * bpp+1],
                                      pixels_top[(start4) * bpp+1], pixels_top[(start5) * bpp+1],
                                      pixels_top[(start6) * bpp+1], pixels_top[(start7) * bpp+1]);
      
      __m256i v_top_b = _mm256_setr_epi32(pixels_top[start * bpp+2], pixels_top[(start1) * bpp+2],
                                      pixels_top[(start2) * bpp+2], pixels_top[(start3) * bpp+2],
                                      pixels_top[(start4) * bpp+2], pixels_top[(start5) * bpp+2],
                                      pixels_top[(start6) * bpp+2], pixels_top[(start7) * bpp+2]);
        
      __m256i cond = v_is_green(v_top_r, v_top_g, v_top_b);

      __m256i v_bg_r = _mm256_setr_epi32(pixels_bg[start * bpp], pixels_bg[(start1) * bpp],
                                      pixels_bg[(start2) * bpp], pixels_bg[(start3) * bpp],
                                      pixels_bg[(start4) * bpp], pixels_bg[(start5) * bpp],
                                      pixels_bg[(start6) * bpp], pixels_bg[(start7) * bpp]);

      __m256i v_bg_g = _mm256_setr_epi32(pixels_bg[start * bpp+1], pixels_bg[(start1) * bpp+1],
                                      pixels_bg[(start2) * bpp+1], pixels_bg[(start3) * bpp+1],
                                      pixels_bg[(start4) * bpp+1], pixels_bg[(start5) * bpp+1],
                                      pixels_bg[(start6) * bpp+1], pixels_bg[(start7) * bpp+1]);

      __m256i v_bg_b = _mm256_setr_epi32(pixels_bg[start * bpp+2], pixels_bg[(start1) * bpp+2],
                                      pixels_bg[(start2) * bpp+2], pixels_bg[(start3) * bpp+2],
                                      pixels_bg[(start4) * bpp+2], pixels_bg[(start5) * bpp+2],
                                      pixels_bg[(start6) * bpp+2], pixels_bg[(start7) * bpp+2]);
        

      v_top_r = _mm256_or_si256(_mm256_and_si256(v_bg_r, cond), _mm256_andnot_si256(cond, v_top_r));
      v_top_g = _mm256_or_si256(_mm256_and_si256(v_bg_g, cond), _mm256_andnot_si256(cond, v_top_g));
      v_top_b = _mm256_or_si256(_mm256_and_si256(v_bg_b, cond), _mm256_andnot_si256(cond, v_top_b));
      
      int temp_r[8];
      int temp_g[8];
      int temp_b[8];

      _mm256_maskstore_epi32(temp_r, _mm256_set1_epi32(~0), v_top_r);
      _mm256_maskstore_epi32(temp_g, _mm256_set1_epi32(~0), v_top_g);
      _mm256_maskstore_epi32(temp_b, _mm256_set1_epi32(~0), v_top_b);

      for(int k=0; k<8; k++){
        pixels_top[(start+k*width) * bpp]=temp_r[k] & 255;
        pixels_top[(start+k*width) * bpp + 1]=temp_g[k] & 255;
        pixels_top[(start+k*width) * bpp + 2]=temp_b[k] & 255;
      }
      
    }
  }

  // perform sharpening - rightwards convolution
  for (int i = 0; i < height; i+=8){
    for (int j = 1; j < width-1; j ++){
      for (int k = 0; k < 3; k++){
        int start = i * width + j;
        int start1 = start + width;
        int start2 = start1 + width;
        int start3 = start2 + width;
        int start4 = start3 + width;
        int start5 = start4 + width;
        int start6 = start5 + width;
        int start7 = start6 + width;      

        // load a window (3x8)  
        __m256i v_top_prev = _mm256_setr_epi32(pixels_top[(start-1) * bpp+k], pixels_top[(start1-1) * bpp+k],
                                        pixels_top[(start2-1) * bpp+k], pixels_top[(start3-1) * bpp+k],
                                        pixels_top[(start4-1) * bpp+k], pixels_top[(start5-1) * bpp+k],
                                        pixels_top[(start6-1) * bpp+k], pixels_top[(start7-1) * bpp+k]);
                                
        __m256i v_top = _mm256_setr_epi32(pixels_top[(start) * bpp+k], pixels_top[(start1) * bpp+k],
                                        pixels_top[(start2) * bpp+k], pixels_top[(start3) * bpp+k],
                                        pixels_top[(start4) * bpp+k], pixels_top[(start5) * bpp+k],
                                        pixels_top[(start6) * bpp+k], pixels_top[(start7) * bpp+k]);
        
        __m256i v_top_next = _mm256_setr_epi32(pixels_top[(start+1) * bpp+k], pixels_top[(start1+1) * bpp+k],
                                        pixels_top[(start2+1) * bpp+k], pixels_top[(start3+1) * bpp+k],
                                        pixels_top[(start4+1) * bpp+k], pixels_top[(start5+1) * bpp+k],
                                        pixels_top[(start6+1) * bpp+k], pixels_top[(start7+1) * bpp+k]);

        
        __m256i weighted_top = _mm256_srai_epi32(v_top, 0);
        __m256i weighted_prev = _mm256_srai_epi32(v_top_prev, 1);
        __m256i weighted_next = _mm256_srai_epi32(v_top_next, 1);

        weighted_top = _mm256_sub_epi32(weighted_top, weighted_prev);
        weighted_top = _mm256_sub_epi32(weighted_top, weighted_next);

        

        v_top = _mm256_add_epi32(weighted_top, v_top);

        v_top = _mm256_min_epi32(v_top, _mm256_set1_epi32(255));
        v_top = _mm256_max_epi32(v_top, _mm256_set1_epi32(0));

        int temp[8];
        
        _mm256_maskstore_epi32(temp, _mm256_set1_epi32(~0), v_top);

        for(int l=0; l<8; l++){
          pixels_top[(start+l*width) * bpp + k] = temp[l]& 255;
        }
      }
    }
  }

  // perform sharpening - downwards convolution
  for (int i = 1; i < height-1; i+=1){
    for (int j = 0; j < width; j += 8){
      for (int k = 0; k < 3; k++){
        
        int start = i * width + j;
        int start1 = start + 1;
        int start2 = start1 + 1;
        int start3 = start2 + 1;
        int start4 = start3 + 1;
        int start5 = start4 + 1;
        int start6 = start5 + 1;
        int start7 = start6 + 1;      

        // load a window (3x8)  
        __m256i v_top_prev = _mm256_setr_epi32(pixels_top[(start-width) * bpp+k], pixels_top[(start1-width) * bpp+k],
                                        pixels_top[(start2-width) * bpp+k], pixels_top[(start3-width) * bpp+k],
                                        pixels_top[(start4-width) * bpp+k], pixels_top[(start5-width) * bpp+k],
                                        pixels_top[(start6-width) * bpp+k], pixels_top[(start7-width) * bpp+k]);
                                
        __m256i v_top = _mm256_setr_epi32(pixels_top[(start) * bpp+k], pixels_top[(start1) * bpp+k],
                                        pixels_top[(start2) * bpp+k], pixels_top[(start3) * bpp+k],
                                        pixels_top[(start4) * bpp+k], pixels_top[(start5) * bpp+k],
                                        pixels_top[(start6) * bpp+k], pixels_top[(start7) * bpp+k]);
        
        __m256i v_top_next = _mm256_setr_epi32(pixels_top[(start+width) * bpp+k], pixels_top[(start1+width) * bpp+k],
                                        pixels_top[(start2+width) * bpp+k], pixels_top[(start3+width) * bpp+k],
                                        pixels_top[(start4+width) * bpp+k], pixels_top[(start5+width) * bpp+k],
                                        pixels_top[(start6+width) * bpp+k], pixels_top[(start7+width) * bpp+k]);

        
        __m256i weighted_top = _mm256_srai_epi32(v_top, 0);
        __m256i weighted_prev = _mm256_srai_epi32(v_top_prev, 1);
        __m256i weighted_next = _mm256_srai_epi32(v_top_next, 1);

        weighted_top = _mm256_sub_epi32(weighted_top, weighted_prev);
        weighted_top = _mm256_sub_epi32(weighted_top, weighted_next);

        

        v_top = _mm256_add_epi32(weighted_top, v_top);

        v_top = _mm256_min_epi32(v_top, _mm256_set1_epi32(255));
        v_top = _mm256_max_epi32(v_top, _mm256_set1_epi32(0));

        int temp[8];
        
        _mm256_maskstore_epi32(temp, _mm256_set1_epi32(~0), v_top);

        for(int l=0; l<8; l++){
          pixels_top[(start+l) * bpp + k] = temp[l]& 255;
        }
      }
    }
  }
  
  
  /* write new image */
  WriteImage("replaced.bmp", pixels_top, width, height, bpp);
  
  /* free everything */
  free(pixels_top);
  free(pixels_bg);
  return 0;
}
