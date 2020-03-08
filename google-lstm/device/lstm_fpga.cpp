#include "lstm_fpga.h"
#include "compute_gate.cpp"
#include "compute_yt.cpp"
#include "compute_vector.cpp"

using namespace std;

void update_xyt( my_fixed xyt[BLOCK_NUM_QXR][FFT_SIZE],
                 my_fixed yt[BLOCK_NUM_PY][FFT_SIZE] ){
#pragma HLS INLINE off
    add_yt_to_xyt: for (unsigned i = 0; i < BLOCK_NUM_QR; ++i) {
#pragma HLS PIPELINE
        for (unsigned k = 0; k < FFT_SIZE; ++k) {
            xyt[i+BLOCK_NUM_QX][k] = yt[i][k];
        }
    }
}

extern "C" {
void lstm( my_fixed * in,
           my_fixed * out ) {   // in & out are all real numbers
// memory interface specification
#pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem

// control interface specification
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // compute_gate matrix
    my_fixed Wi_real[W_PXR_SIZE][W_QXR_SIZE], Wi_imag[W_PXR_SIZE][W_QXR_SIZE];
    my_fixed Wf_real[W_PXR_SIZE][W_QXR_SIZE], Wf_imag[W_PXR_SIZE][W_QXR_SIZE];
    my_fixed Wc_real[W_PXR_SIZE][W_QXR_SIZE], Wc_imag[W_PXR_SIZE][W_QXR_SIZE];
    my_fixed Wo_real[W_PXR_SIZE][W_QXR_SIZE], Wo_imag[W_PXR_SIZE][W_QXR_SIZE];
#pragma HLS ARRAY_PARTITION variable=Wi_real cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wi_imag cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wf_real cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wf_imag cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wc_real cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wc_imag cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wo_real cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wo_imag cyclic factor=16 dim=1 partition

#pragma HLS ARRAY_PARTITION variable=Wi_real cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wi_imag cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wf_real cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wf_imag cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wc_real cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wc_imag cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wo_real cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wo_imag cyclic factor=5 dim=2 partition

    // compute_vector vetor
    my_fixed Vi[BLOCK_NUM_QY][FFT_SIZE];
    my_fixed Vf[BLOCK_NUM_QY][FFT_SIZE];                     
    my_fixed Vo[BLOCK_NUM_QY][FFT_SIZE];                     
    my_fixed bi[BLOCK_NUM_QY][FFT_SIZE];                     
    my_fixed bf[BLOCK_NUM_QY][FFT_SIZE]; 
    my_fixed bc[BLOCK_NUM_QY][FFT_SIZE];                     
    my_fixed bo[BLOCK_NUM_QY][FFT_SIZE];

    // compute_yt matrix
    my_fixed Wy_real[W_PY_SIZE][W_QY_SIZE], Wy_imag[W_PY_SIZE][W_QY_SIZE];
#pragma HLS ARRAY_PARTITION variable=Wy_real cyclic factor=16 dim=1 partition
#pragma HLS ARRAY_PARTITION variable=Wy_imag cyclic factor=16 dim=1 partition

#pragma HLS ARRAY_PARTITION variable=Wy_real cyclic factor=5 dim=2 partition
#pragma HLS ARRAY_PARTITION variable=Wy_imag cyclic factor=5 dim=2 partition

    // gate compute stage output data buffer
    my_fixed  it_buf1[BLOCK_NUM_PXR][FFT_SIZE] = {0}, it_buf2[BLOCK_NUM_PXR][FFT_SIZE] = {0};
    my_fixed  ft_buf1[BLOCK_NUM_PXR][FFT_SIZE] = {0}, ft_buf2[BLOCK_NUM_PXR][FFT_SIZE] = {0};
    my_fixed  gt_buf1[BLOCK_NUM_PXR][FFT_SIZE] = {0}, gt_buf2[BLOCK_NUM_PXR][FFT_SIZE] = {0};
    my_fixed  ot_buf1[BLOCK_NUM_PXR][FFT_SIZE] = {0}, ot_buf2[BLOCK_NUM_PXR][FFT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=it_buf1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=ft_buf1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=gt_buf1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=ot_buf1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=it_buf2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=ft_buf2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=gt_buf2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=ot_buf2 complete dim=2

    // ct 
    my_fixed  ct[BLOCK_NUM_QY][FFT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=ct complete dim=2

    // mt
    my_fixed  mt_buf1[BLOCK_NUM_QY][FFT_SIZE] = {0}, mt_buf2[BLOCK_NUM_QY][FFT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=mt_buf1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=mt_buf2 complete dim=2

    // xt
    my_fixed xt[BLOCK_NUM_QX][FFT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=xt complete dim=2

    // yt
    my_fixed yt_buf1[BLOCK_NUM_PY][FFT_SIZE] = {0}, yt_buf2[BLOCK_NUM_PY][FFT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=yt_buf1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=yt_buf2 complete dim=2

    // xyt
    my_fixed xyt_buf1_1[BLOCK_NUM_QXR][FFT_SIZE] = {0}, xyt_buf2_1[BLOCK_NUM_QXR][FFT_SIZE] = {0};  
    my_fixed xyt_buf1_2[BLOCK_NUM_QXR][FFT_SIZE] = {0}, xyt_buf2_2[BLOCK_NUM_QXR][FFT_SIZE] = {0};  
    my_fixed xyt_buf1_3[BLOCK_NUM_QXR][FFT_SIZE] = {0}, xyt_buf2_3[BLOCK_NUM_QXR][FFT_SIZE] = {0};  
    my_fixed xyt_buf1_4[BLOCK_NUM_QXR][FFT_SIZE] = {0}, xyt_buf2_4[BLOCK_NUM_QXR][FFT_SIZE] = {0};  
#pragma HLS ARRAY_PARTITION variable=xyt_buf1_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=xyt_buf2_1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=xyt_buf1_2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=xyt_buf2_2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=xyt_buf1_3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=xyt_buf2_3 complete dim=2
#pragma HLS ARRAY_PARTITION variable=xyt_buf1_4 complete dim=2
#pragma HLS ARRAY_PARTITION variable=xyt_buf2_4 complete dim=2

    // load data to xt
    load_data_to_xt: for (unsigned j = 0; j < BLOCK_NUM_QX; ++j) {
        memcpy(&xt[j], (my_fixed *)(in+j*FFT_SIZE), FFT_SIZE);
    }
    // union data to xyt
    add_xt_to_xyt: for (unsigned i = 0; i < BLOCK_NUM_QX; ++i) {
#pragma HLS PIPELINE
        for (unsigned k = 0; k < FFT_SIZE; ++k) {
            xyt_buf1_1[i][k] = xt[i][k];
            xyt_buf2_1[i][k] = xt[i][k];
            xyt_buf1_2[i][k] = xt[i][k];
            xyt_buf2_2[i][k] = xt[i][k];
            xyt_buf1_3[i][k] = xt[i][k];
            xyt_buf2_3[i][k] = xt[i][k];
            xyt_buf1_4[i][k] = xt[i][k];
            xyt_buf2_4[i][k] = xt[i][k];
        }
    }


    // lsmt main loop
    main_lstm_loop: for (unsigned t = 0; t < 100; ++t) {

        if ( t%2 == 0 ) {
            // union data to xyt
            update_xyt(xyt_buf1_1, yt_buf2);
            update_xyt(xyt_buf1_2, yt_buf2);
            update_xyt(xyt_buf1_3, yt_buf2);
            update_xyt(xyt_buf1_4, yt_buf2);

            // stage 1
            compute_gate(xyt_buf2_1, Wi_real, Wi_imag, it_buf1); // it
            compute_gate(xyt_buf2_2, Wf_real, Wf_imag, ft_buf1); // ft
            compute_gate(xyt_buf2_3, Wc_real, Wc_imag, gt_buf1); // gt
            compute_gate(xyt_buf2_4, Wo_real, Wo_imag, ot_buf1); // ot

            // stage 2
            compute_vector(it_buf2, ft_buf2, gt_buf2, ot_buf2, ct, mt_buf1, Vi, Vf, Vo, bi, bf, bc, bo);

            // stage 3
            compute_yt(mt_buf2, Wy_real, Wy_imag, yt_buf1);
        }
        else {
            // union data to xyt
            update_xyt(xyt_buf2_1, yt_buf1);
            update_xyt(xyt_buf2_2, yt_buf1);
            update_xyt(xyt_buf2_3, yt_buf1);
            update_xyt(xyt_buf2_4, yt_buf1);

            // stage 1
            compute_gate(xyt_buf1_1, Wi_real, Wi_imag, it_buf2); // it
            compute_gate(xyt_buf1_2, Wf_real, Wf_imag, ft_buf2); // ft
            compute_gate(xyt_buf1_3, Wc_real, Wc_imag, gt_buf2); // gt
            compute_gate(xyt_buf1_4, Wo_real, Wo_imag, ot_buf2); // ot

            // stage 2
            compute_vector(it_buf1, ft_buf1, gt_buf1, ot_buf1, ct, mt_buf2, Vi, Vf, Vo, bi, bf, bc, bo);

            // stage 3
            compute_yt(mt_buf1, Wy_real, Wy_imag, yt_buf2);

        }
    }
    

    loop_write_output_data: for (unsigned i = 0; i < BLOCK_NUM_PY; ++i) {
        memcpy((out+i*FFT_SIZE), (my_fixed *)(&yt_buf2[i]), FFT_SIZE);
    }
    return;
}
}
