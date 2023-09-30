# Q5

## Q5-a

We used 'Unified Memory' in our code which you can learn about [here](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

```c
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define n 100
#define m 100
#define tol 0.01

using namespace std;

double diffmax;

// Kernel function
__global__
void updateTemp(double **t, double **tnew, double **diff)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (1 <= i && i <= m && 1 <= j && j <= n) {
    tnew[i][j] = (t[i - 1][j] + t[i + 1][j] + t[i][j - 1] + t[i][j + 1]) / 4.0;
    diff[i][j] = fabs(tnew[i][j] - t[i][j]);
    
    // copy new to old temperatures
    t[i][j] = tnew[i][j];
  }
}

int main(void)
{
  struct timeval startTime, stopTime;
  long totalTime;
  double **t, **tnew, **diff;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&t, (m + 2)*sizeof(double*));
  for (int i = 0; i < m + 2; ++i)
    cudaMallocManaged(&t[i], (n + 2)*sizeof(double));
  cudaMallocManaged(&tnew, (m + 2)*sizeof(double*));
  for (int i = 0; i < m + 2; ++i)
    cudaMallocManaged(&tnew[i], (n + 2)*sizeof(double));
  cudaMallocManaged(&diff, (m + 2)*sizeof(double*));
  for (int i = 0; i < m + 2; ++i)
    cudaMallocManaged(&diff[i], (n + 2)*sizeof(double));

  for (int z = 0; z < 11; ++z) {
    gettimeofday(&startTime, NULL);

    // initialize x and y arrays on the host
    for (int i = 0; i < m + 2; ++i)
      for (int j = 0; j < n + 2; ++j)
        t[i][j] = 30.0;
    // fix boundary conditions
    for (int i = 1; i <= m; ++i) {
      t[i][0] = 10.0;
      t[i][n + 1] = 140.0;
    }
    for (int j = 1; j <= n; ++j) {
      t[0][j] = 20.0;
      t[m + 1][j] = 100.0;
    }

    // main loop
    int iter = 0;
    diffmax = 1000000.0;
    dim3 gd(m);
    dim3 bd(n);
    
    while (diffmax > tol) {
      ++iter;

      // update temperature for next iteration
      // Run kernel on 1M elements on the GPU
      updateTemp<<<gd, bd>>>(t, tnew, diff);

      // Wait for GPU to finish before accessing on host
      cudaDeviceSynchronize();

      // work out maximum difference between old and new temperatures
      diffmax = 0.0;
      for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j)
          if (diff[i][j] > diffmax)
            diffmax = diff[i][j];
    }

    gettimeofday(&stopTime, NULL);
    totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) -
                (startTime.tv_sec * 1000000 + startTime.tv_usec);

    printf("%ld\n", totalTime);
  }
  // Free memory
  for (int i = 0; i < m + 2; ++i) {
    cudaFree(t[i]);
    cudaFree(tnew[i]);
    cudaFree(diff[i]);
  }
  cudaFree(t);
  cudaFree(tnew);
  cudaFree(diff);

  return 0;
}

//write this code in jacobi.cu then run below commands for complete analyze
//nvcc jacobi.cu -o jacobi_cuda
//nvprof ./jacobi_cuda
```

## Q5-b

![chart](plot.png)

## Q5-c

In OpenMP implementation numbers were in scale of milli seconds. Using CUDA numbers are in scale of micro seconds. The differance is huge.

## Q5-d

We explained two reasons for our results on OpenMP implementation.

Considering those problems with OpenMP, CUDA (using GPUs) solves the problems we had and significantly reduces execution time. 
