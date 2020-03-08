#include "lstm_fpga.h"

//extern "C" {
void fft8_fixed(my_fixed x0_real[FFT_SIZE], my_fixed y0_real[FFT_SIZE], my_fixed y0_imag[FFT_SIZE]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=x0_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=y0_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=y0_imag complete dim=1

#pragma HLS PIPELINE II=1

    // fft twiddle values
    my_fixed  t1_real[FFT_SIZE/2]  = {  1.0000,              // stage 1
                                       0.7071, 
                                       0.0000,
                                      -0.7071 };

    my_fixed  t1_imag[FFT_SIZE/2]  = {  0.0000,              // stage 1
                                      -0.7071, 
                                      -1.0000,
                                      -0.7071 };


    my_fixed  t2_real[FFT_SIZE/4]  = {  1.0000,              // stage 2
                                       0.0000 };

    my_fixed  t2_imag[FFT_SIZE/4]  = {  0.0000,              // stage 2
                                      -1.0000 };


    my_fixed  t3_real[FFT_SIZE/8] = { 1.0000};              // stage 3

    my_fixed  t3_imag[FFT_SIZE/8] = { 0.0000};              // stage 3

#pragma HLS ARRAY_PARTITION variable=t1_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=t2_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=t3_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=t1_imag complete dim=1
#pragma HLS ARRAY_PARTITION variable=t2_imag complete dim=1
#pragma HLS ARRAY_PARTITION variable=t3_imag complete dim=1

    // stage data
    my_fixed  x1_real[FFT_SIZE], x1_imag[FFT_SIZE]; 
    my_fixed  x2_real[FFT_SIZE], x2_imag[FFT_SIZE];
    my_fixed  x3_real[FFT_SIZE], x3_imag[FFT_SIZE];
#pragma HLS ARRAY_PARTITION variable=x1_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=x2_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=x3_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=x1_imag complete dim=1
#pragma HLS ARRAY_PARTITION variable=x2_imag complete dim=1
#pragma HLS ARRAY_PARTITION variable=x3_imag complete dim=1

    // stage 1
    loop_stage_1: for (unsigned m = 0; m < 4; ++m) {
        x1_real[m]   = x0_real[m] + x0_real[m+4];
        x1_imag[m]   = 0.0000;

        my_fixed d1_real = x0_real[m] - x0_real[m+4];
        x1_real[m+4] = d1_real * t1_real[m];
        x1_imag[m+4] = d1_real * t1_imag[m];
    }

    // stage 2
    loop_stage_2: for (unsigned m = 0; m < 2; ++m) {
        for (unsigned d = 0; d < 8; d+=4) {
            x2_real[m+d]   = x1_real[m+d] + x1_real[m+d+2];
            x2_imag[m+d]   = x1_imag[m+d] + x1_imag[m+d+2];

            my_fixed d2_real = x1_real[m+d] - x1_real[m+d+2];
            my_fixed d2_imag = x1_imag[m+d] - x1_imag[m+d+2];
            x2_real[m+d+2] = d2_real * t2_real[m] - d2_imag * t2_imag[m];
            x2_imag[m+d+2] = d2_real * t2_imag[m] + d2_imag * t2_real[m];
        }
    }

    // stage 3
    loop_stage_3: for (unsigned d = 0; d < 8; d+=2) {
            x3_real[d]   = x2_real[d] + x2_real[d+1];
            x3_imag[d]   = x2_imag[d] + x2_imag[d+1];

            my_fixed d3_real = x2_real[d] - x2_real[d+1];
            my_fixed d3_imag = x2_imag[d] - x2_imag[d+1];
            x3_real[d+1] = d3_real * t3_real[0] - d3_imag * t3_imag[0];
            x3_imag[d+1] = d3_real * t3_imag[0] + d3_imag * t3_real[0];
    }

    // bit reversal
    y0_real[0] = x3_real[0];
    y0_real[1] = x3_real[4];
    y0_real[2] = x3_real[2];
    y0_real[3] = x3_real[6];
    y0_real[4] = x3_real[1];
    y0_real[5] = x3_real[5];
    y0_real[6] = x3_real[3];
    y0_real[7] = x3_real[7];

    y0_imag[0] = x3_imag[0];
    y0_imag[1] = x3_imag[4];
    y0_imag[2] = x3_imag[2];
    y0_imag[3] = x3_imag[6];
    y0_imag[4] = x3_imag[1];
    y0_imag[5] = x3_imag[5];
    y0_imag[6] = x3_imag[3];
    y0_imag[7] = x3_imag[7];
}
//}
