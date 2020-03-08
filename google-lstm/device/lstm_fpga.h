#ifndef _FFT8_FPGA_H_
#define _FFT8_FPGA_H_

#include "math.h"
#include "ap_fixed.h"
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <string>  
#include <vector>  
#include <fstream>  
#include <sstream>  
#include <cstdlib>
#include <stdlib.h>

// fft size
#define FFT_SIZE 8

// weight matirx block numbers
#define BLOCK_NUM_PX 128 // 1024 / 8 = 128
#define BLOCK_NUM_QX 20 // (int)(153 / 8.0 + 0.5)
#define BLOCK_NUM_PR 128 // 1024 / 8 = 128
#define BLOCK_NUM_QR 64 // 512 / 8 = 64 
#define BLOCK_NUM_PXR (BLOCK_NUM_PX) // 128
#define BLOCK_NUM_QXR (BLOCK_NUM_QX + BLOCK_NUM_QR) // 20 + 64 = 84

#define BLOCK_NUM_PY 64 // 512 / 8 = 64
#define BLOCK_NUM_QY 128 // 1024 / 8 = 128

// weight matrix sizes
#define W_PXR_SIZE BLOCK_NUM_PXR // 128
#define W_QXR_SIZE ((FFT_SIZE-3) * BLOCK_NUM_QXR) // 672*5/8

#define W_PY_SIZE BLOCK_NUM_PY // 64
#define W_QY_SIZE ((FFT_SIZE-3) * BLOCK_NUM_QY) // 1024*5/8

// parallelism
#define PLP_XR 16  // P Level Parallelism for input
#define PLP_V 2   // compute vector Parallelism
#define PLP_Y 16   // P Level Parallelism for input


// fixed add latency for add accumulation
#define FIXED_ADD_LAT 2



typedef ap_fixed<16,8> my_fixed;
//typedef double my_fixed;


/*
void fft8_fixed ( fixed x0_real[FFT_SIZE],
                   fixed y0_real[FFT_SIZE],
                   fixed y0_imag[FFT_SIZE] );

void ifft8_fixed( fixed x0_real[FFT_SIZE], 
                   fixed x0_imag[FFT_SIZE],
                   fixed y0_real[FFT_SIZE] );
*/

void compute_gate( my_fixed x0_real[BLOCK_NUM_QXR][FFT_SIZE],
                   //const fixed w0_real[W_PXR_SIZE][W_QXR_SIZE],
                   //const fixed w0_imag[W_PXR_SIZE][W_QXR_SIZE],
                   my_fixed w0_real[W_PXR_SIZE][W_QXR_SIZE],
                   my_fixed w0_imag[W_PXR_SIZE][W_QXR_SIZE],
                   my_fixed y0_real[BLOCK_NUM_PXR][FFT_SIZE]);

void compute_yt(   my_fixed x0_real[BLOCK_NUM_QY][FFT_SIZE],
                   //const fixed w0_real[W_PY_SIZE][W_QY_SIZE],
                   //const fixed w0_imag[W_PY_SIZE][W_QY_SIZE],
                   my_fixed w0_real[W_PY_SIZE][W_QY_SIZE],
                   my_fixed w0_imag[W_PY_SIZE][W_QY_SIZE],
                   my_fixed y0_real[BLOCK_NUM_PY][FFT_SIZE]);

//#include "lstm_matrix_rom.cpp"
//#include "compute_gate_golden_result.cpp"
//#include "lstm_vector_rom.cpp"
//#include "fft8_fpga.cpp"
//#include "ifft8_fpga.cpp"
//#include "compute_gate.cpp"
//#include "activation_func.cpp"
//#include "compute_vector.cpp"
//#include "compute_yt.cpp"


#endif
