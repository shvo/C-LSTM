#include "lstm_fpga.h"
//#include "fft8_fpga.cpp"
//#include "ifft8_fpga.cpp"

void compute_yt( my_fixed fft_in_buf_real[BLOCK_NUM_QY][FFT_SIZE],        // without ifft
                 my_fixed W_real[W_PY_SIZE][W_QY_SIZE],
                 my_fixed W_imag[W_PY_SIZE][W_QY_SIZE],
                 my_fixed ifft_out_buf_real[BLOCK_NUM_PY][FFT_SIZE]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=fft_in_buf_real complete dim=2
#pragma HLS ARRAY_PARTITION variable=ifft_out_buf_real complete dim=2

// partition rom data
#pragma HLS ARRAY_PARTITION variable=W_real cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=W_imag cyclic factor=16 dim=1 partition

#pragma HLS ARRAY_PARTITION variable=W_real cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=W_imag cyclic factor=5 dim=2 partition

    // fft output data buffer (complex number)
    my_fixed  fft_out_buf_real[FIXED_ADD_LAT][FFT_SIZE];   
    my_fixed  fft_out_buf_imag[FIXED_ADD_LAT][FFT_SIZE];   
#pragma HLS ARRAY_PARTITION variable=fft_out_buf_real complete dim=0
#pragma HLS ARRAY_PARTITION variable=fft_out_buf_imag complete dim=0

    // add accumulation buffer
    my_fixed sum_real_p[FIXED_ADD_LAT][BLOCK_NUM_PY][FFT_SIZE-3] = {0};
    my_fixed sum_imag_p[FIXED_ADD_LAT][BLOCK_NUM_PY][FFT_SIZE-3] = {0};
#pragma HLS ARRAY_PARTITION variable=sum_real_p complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum_imag_p complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum_real_p cyclic factor=16 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=sum_imag_p cyclic factor=16 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=sum_real_p complete dim=3
#pragma HLS ARRAY_PARTITION variable=sum_imag_p complete dim=3

    // ifft input buffer
    my_fixed ifft_in_buf_real[FFT_SIZE];
    my_fixed ifft_in_buf_imag[FFT_SIZE];
#pragma HLS ARRAY_PARTITION variable=ifft_in_buf_real complete dim=1
#pragma HLS ARRAY_PARTITION variable=ifft_in_buf_imag complete dim=1

/*
loop_sum_p_init: for (unsigned i = 0; i < BLOCK_NUM_PY; i+=PLP_Y) {
#pragma HLS PIPELINE II=1
    for (unsigned p = 0; p < PLP_Y; ++p) {
        sum_real_p[0][i+p][0] = 0;
        sum_imag_p[0][i+p][0] = 0;
        sum_real_p[1][i+p][0] = 0;
        sum_imag_p[1][i+p][0] = 0;

        sum_real_p[0][i+p][1] = 0;
        sum_imag_p[0][i+p][1] = 0;
        sum_real_p[1][i+p][1] = 0;
        sum_imag_p[1][i+p][1] = 0;

        sum_real_p[0][i+p][2] = 0;
        sum_imag_p[0][i+p][2] = 0;
        sum_real_p[1][i+p][2] = 0;
        sum_imag_p[1][i+p][2] = 0;

        sum_real_p[0][i+p][3] = 0;
        sum_imag_p[0][i+p][3] = 0;
        sum_real_p[1][i+p][3] = 0;
        sum_imag_p[1][i+p][3] = 0;

        sum_real_p[0][i+p][4] = 0;
        sum_imag_p[0][i+p][4] = 0;
        sum_real_p[1][i+p][4] = 0;
        sum_imag_p[1][i+p][4] = 0;
    }
}
*/


loop_lstm_i: for (unsigned i = 0; i < BLOCK_NUM_PY; i+=PLP_Y) {
    loop_lstm_j: for (unsigned j = 0; j < BLOCK_NUM_QY; j+=FIXED_ADD_LAT) {
#pragma HLS PIPELINE II=2

        // perform 16-input fft
        fft8_fixed(fft_in_buf_real[j+0], fft_out_buf_real[0], fft_out_buf_imag[0]);
        fft8_fixed(fft_in_buf_real[j+1], fft_out_buf_real[1], fft_out_buf_imag[1]);

        loop_mul_plp: for (unsigned p = 0; p < PLP_Y; ++p) {
            // perform point-wise multiplication and accumulation
            sum_real_p[0][i+p][0] += fft_out_buf_real[0][0] * W_real[i+p][(j+0)*(FFT_SIZE-3)+0] - fft_out_buf_imag[0][0] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+0];
            sum_imag_p[0][i+p][0] += fft_out_buf_real[0][0] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+0] + fft_out_buf_imag[0][0] * W_real[i+p][(j+0)*(FFT_SIZE-3)+0];
            sum_real_p[1][i+p][0] += fft_out_buf_real[1][0] * W_real[i+p][(j+1)*(FFT_SIZE-3)+0] - fft_out_buf_imag[1][0] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+0];
            sum_imag_p[1][i+p][0] += fft_out_buf_real[1][0] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+0] + fft_out_buf_imag[1][0] * W_real[i+p][(j+1)*(FFT_SIZE-3)+0];

            sum_real_p[0][i+p][1] += fft_out_buf_real[0][1] * W_real[i+p][(j+0)*(FFT_SIZE-3)+1] - fft_out_buf_imag[0][1] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+1];
            sum_real_p[0][i+p][1] += fft_out_buf_real[0][1] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+1] + fft_out_buf_imag[0][1] * W_real[i+p][(j+0)*(FFT_SIZE-3)+1];
            sum_real_p[1][i+p][1] += fft_out_buf_real[1][1] * W_real[i+p][(j+1)*(FFT_SIZE-3)+1] - fft_out_buf_imag[1][1] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+1];
            sum_real_p[1][i+p][1] += fft_out_buf_real[1][1] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+1] + fft_out_buf_imag[1][1] * W_real[i+p][(j+1)*(FFT_SIZE-3)+1];

            sum_real_p[0][i+p][2] += fft_out_buf_real[0][2] * W_real[i+p][(j+0)*(FFT_SIZE-3)+2] - fft_out_buf_imag[0][2] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+2];
            sum_real_p[0][i+p][2] += fft_out_buf_real[0][2] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+2] + fft_out_buf_imag[0][2] * W_real[i+p][(j+0)*(FFT_SIZE-3)+2];
            sum_real_p[1][i+p][2] += fft_out_buf_real[1][2] * W_real[i+p][(j+1)*(FFT_SIZE-3)+2] - fft_out_buf_imag[1][2] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+2];
            sum_real_p[1][i+p][2] += fft_out_buf_real[1][2] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+2] + fft_out_buf_imag[1][2] * W_real[i+p][(j+1)*(FFT_SIZE-3)+2];

            sum_real_p[0][i+p][3] += fft_out_buf_real[0][3] * W_real[i+p][(j+0)*(FFT_SIZE-3)+3] - fft_out_buf_imag[0][3] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+3];
            sum_real_p[0][i+p][3] += fft_out_buf_real[0][3] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+3] + fft_out_buf_imag[0][3] * W_real[i+p][(j+0)*(FFT_SIZE-3)+3];
            sum_real_p[1][i+p][3] += fft_out_buf_real[1][3] * W_real[i+p][(j+1)*(FFT_SIZE-3)+3] - fft_out_buf_imag[1][3] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+3];
            sum_real_p[1][i+p][3] += fft_out_buf_real[1][3] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+3] + fft_out_buf_imag[1][3] * W_real[i+p][(j+1)*(FFT_SIZE-3)+3];

            sum_real_p[0][i+p][4] += fft_out_buf_real[0][4] * W_real[i+p][(j+0)*(FFT_SIZE-3)+4] - fft_out_buf_imag[0][4] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+4];
            sum_real_p[0][i+p][4] += fft_out_buf_real[0][4] * W_imag[i+p][(j+0)*(FFT_SIZE-3)+4] + fft_out_buf_imag[0][4] * W_real[i+p][(j+0)*(FFT_SIZE-3)+4];
            sum_real_p[1][i+p][4] += fft_out_buf_real[1][4] * W_real[i+p][(j+1)*(FFT_SIZE-3)+4] - fft_out_buf_imag[1][4] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+4];
            sum_real_p[1][i+p][4] += fft_out_buf_real[1][4] * W_imag[i+p][(j+1)*(FFT_SIZE-3)+4] + fft_out_buf_imag[1][4] * W_real[i+p][(j+1)*(FFT_SIZE-3)+4];
        }
    }
}

    // perform sum final
    loop_sum_p: for (unsigned i = 0; i < BLOCK_NUM_PY; ++i) {
#pragma HLS PIPELINE II=1

        // conj & divided by FFT_SIZE
        ifft_in_buf_real[0] = (sum_real_p[0][i][0] + sum_real_p[1][i][0]) / FFT_SIZE;
        ifft_in_buf_imag[0] = (sum_imag_p[0][i][0] + sum_imag_p[1][i][0]) / FFT_SIZE;
        ifft_in_buf_real[1] = (sum_real_p[0][i][1] + sum_real_p[1][i][1]) / FFT_SIZE;
        ifft_in_buf_imag[1] = (sum_imag_p[0][i][1] + sum_imag_p[1][i][1]) / FFT_SIZE;
        ifft_in_buf_real[2] = (sum_real_p[0][i][2] + sum_real_p[1][i][2]) / FFT_SIZE;
        ifft_in_buf_imag[2] = (sum_imag_p[0][i][2] + sum_imag_p[1][i][2]) / FFT_SIZE;
        ifft_in_buf_real[3] = (sum_real_p[0][i][3] + sum_real_p[1][i][3]) / FFT_SIZE;
        ifft_in_buf_imag[3] = (sum_imag_p[0][i][3] + sum_imag_p[1][i][3]) / FFT_SIZE;
        ifft_in_buf_real[4] = (sum_real_p[0][i][4] + sum_real_p[1][i][4]) / FFT_SIZE;
        ifft_in_buf_imag[4] = (sum_imag_p[0][i][4] + sum_imag_p[1][i][4]) / FFT_SIZE;

        ifft_in_buf_real[5] = ifft_in_buf_real[3];
        ifft_in_buf_real[6] = ifft_in_buf_real[2];
        ifft_in_buf_real[7] = ifft_in_buf_real[1];

        ifft_in_buf_imag[5] = ifft_in_buf_imag[3];
        ifft_in_buf_imag[6] = ifft_in_buf_imag[2];
        ifft_in_buf_imag[7] = ifft_in_buf_imag[1];

        ifft_in_buf_imag[0] = -ifft_in_buf_imag[0];
        ifft_in_buf_imag[1] = -ifft_in_buf_imag[1];
        ifft_in_buf_imag[2] = -ifft_in_buf_imag[2];
        ifft_in_buf_imag[3] = -ifft_in_buf_imag[3];
        ifft_in_buf_imag[4] = -ifft_in_buf_imag[4];

        ifft8_fixed(ifft_in_buf_real, ifft_in_buf_imag, ifft_out_buf_real[i]);
    }
}
