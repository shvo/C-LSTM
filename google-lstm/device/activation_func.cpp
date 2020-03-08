#include "lstm_fpga.h"

// sigmoid
#define SIGMOID_STEP_NUM  20
#define SIGMOID_DRIFT (SIGMOID_STEP_NUM/2)

// tanh
#define TANH_STEP_NUM  20
#define TANH_DRIFT     (TANH_STEP_NUM/2)

// sigmoid
const my_fixed k_sigmoid[SIGMOID_STEP_NUM] = {-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
const my_fixed b_sigmoid[SIGMOID_STEP_NUM] = {-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
const my_fixed sigmoid_step_size = 0.5;

// tanh
const my_fixed k_tanh[TANH_STEP_NUM] = {-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
const my_fixed b_tanh[TANH_STEP_NUM] = {-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
const my_fixed tanh_step_size = 0.5;

void sigmoid ( my_fixed in,
               my_fixed out) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=k_sigmoid complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_sigmoid complete dim=1

    int index = (int)(in/sigmoid_step_size) + SIGMOID_DRIFT;
    if (index < 0) {
        index -= 1;
    }
    out = k_sigmoid[index]*in+b_sigmoid[index];
}

void h ( my_fixed in,
         my_fixed out) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=k_tanh complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_tanh complete dim=1

    int index = (int)(in/tanh_step_size) + TANH_DRIFT;
    if (index < 0) {
        index -= 1;
    }
    out = k_tanh[index]*in+b_tanh[index];
}
