#include "lstm_fpga.h"
#include "activation_func.cpp"

void compute_vector ( my_fixed it_in[BLOCK_NUM_QY][FFT_SIZE],
                      my_fixed ft_in[BLOCK_NUM_QY][FFT_SIZE],
                      my_fixed gt_in[BLOCK_NUM_QY][FFT_SIZE],
                      my_fixed ot_in[BLOCK_NUM_QY][FFT_SIZE],
                      my_fixed ct_in[BLOCK_NUM_QY][FFT_SIZE],
                      my_fixed mt_out[BLOCK_NUM_QY][FFT_SIZE],
                      my_fixed Vi[BLOCK_NUM_QY][FFT_SIZE],
                      my_fixed Vf[BLOCK_NUM_QY][FFT_SIZE],                     
                      my_fixed Vo[BLOCK_NUM_QY][FFT_SIZE],                     
                      my_fixed bi[BLOCK_NUM_QY][FFT_SIZE],                     
                      my_fixed bf[BLOCK_NUM_QY][FFT_SIZE],                     
                      my_fixed bc[BLOCK_NUM_QY][FFT_SIZE],                     
                      my_fixed bo[BLOCK_NUM_QY][FFT_SIZE] ) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=it_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=ft_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=gt_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=ot_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=ct_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=mt_out complete dim=2

    // it buf
    my_fixed it_buf[PLP_V], it_out[PLP_V];
#pragma HLS ARRAY_PARTITION variable=it_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=it_out complete dim=1

    // ft buf
    my_fixed ft_buf[PLP_V], ft_out[PLP_V];
#pragma HLS ARRAY_PARTITION variable=ft_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=ft_out complete dim=1

    // gt buf
    my_fixed gt_buf[PLP_V], gt_out[PLP_V];
#pragma HLS ARRAY_PARTITION variable=gt_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=gt_out complete dim=1

    // ot buf
    my_fixed ot_buf[PLP_V], ot_out[PLP_V];
#pragma HLS ARRAY_PARTITION variable=ot_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=ot_out complete dim=1

    // mt buf
    my_fixed mt_buf[PLP_V]; 
#pragma HLS ARRAY_PARTITION variable=mt_buf complete dim=1

    // ct_in buf
    my_fixed ct_buf[PLP_V]; 
#pragma HLS ARRAY_PARTITION variable=ct_buf complete dim=1


    loop_vector: for (unsigned i = 0; i < BLOCK_NUM_QY; ++i) {
        for (unsigned j = 0; j < FFT_SIZE; j+=PLP_V) {
#pragma HLS PIPELINE II=1
            for (unsigned r = 0; r < PLP_V; ++r) {
                // it
                it_buf[r] = it_in[i][j+r] + Vi[i][j]*ct_in[i][j+r] + bi[i][j+r];
                sigmoid(it_buf[r], it_out[r]);

                // ft
                ft_buf[r] = ft_in[i][j+r] + Vf[i][j+r]*ct_in[i][j+r] + bf[i][j+r];
                sigmoid(ft_buf[r], ft_out[r]);

                // gt
                gt_buf[r] = gt_in[i][j+r] + bc[i][j+r];
                sigmoid(gt_buf[r], gt_out[r]);

                // ct
                //ct_in[i][j+r] = ft_out[r]*ct_in[i][j+r] + gt_out[r]*it_out[r];
                ct_buf[r] = ft_out[r]*ct_in[i][j+r] + gt_out[r]*it_out[r];

                // ot
                //ot_buf[r] = ot_in[i][j+r] + Vo[i][j+r]*ct_in[i][j+r] + bo[i][j+r];
                ot_buf[r] = ot_in[i][j+r] + Vo[i][j+r]*ct_buf[r] + bo[i][j+r];
                sigmoid(ot_buf[r], ot_out[r]);

                // mt
                //h(ct_in[i][j+r], mt_buf[r]); //tanh
                h(ct_buf[r], mt_buf[r]); //tanh
                mt_out[i][j+r] = ot_out[r] * mt_buf[r];
            }
        }
    }
}
