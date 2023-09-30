# Multicore_Computing

### Green Screen
Takes two input images; one as a subject (with a green screen background) and another as a background, and creates an output image with the former's foreground placed on the provided background.

This is available using `AVX2` intrinsics in the `GreenScreen_AVX` folder and can be compiled using the `-mavx2` flag.
A `CUDA` implementation is also available in the `GreenScreen_CUDA` folder which can be run in Google Colab (note that a GPU accelerator must be selected for your workspace).

Also, a sharpening algorithm has been implemented in both versions which can slightly sharpen the output.

### Image Inversion
Inverts an image, implemented as a `CUDA` warmup.

### Jacobi
An implementation of an algorithm sped up using `CUDA`.

### Sobel
An edge-detection implemented using a simple Sobel filter using `CUDA`. Works with high-res (8K) images, and includes brightening and threshold filters. A single-thread implementation is provided to demonstrate speedup.
