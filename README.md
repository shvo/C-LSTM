# C-LSTM

[![GitHub license](https://dmlc.github.io/img/apache2.svg)](./LICENSE)

# C-LSTM: Enabling Efficient LSTM using Structured Compression Techniques on FPGAs

[Introduction](##Introduction) | [Installation](##Installation) | [Tutorial](##Tutorial) | [Publication](##Publication)

## Note
This project is opensource but still in the preliminary building stage. If you have any question, please contact through email: shvowang@pku.edu.cn.

## Introduction
Recently, significant accuracy improvement has been achieved for acoustic recognition systems by increasing the model size of Long Short-Term Memory (LSTM) networks. Unfortunately, the ever- increasing size of LSTM model leads to inefficient designs on FPGAs due to the limited on-chip resources. The previous work proposes to use a pruning based compression technique to reduce the model size and thus speedups the inference on FPGAs. However, the random nature of the pruning technique transforms the dense matrices of the model to highly unstructured sparse ones, which leads to unbalanced computation and irregular memory accesses and thus hurts the overall performance and energy efficiency.

In contrast, we propose to use a structured compression technique which could not only reduce the LSTM model size but also eliminate the irregularities of computation and memory accesses. This approach employs block-circulant instead of sparse matrices to compress weight matrices and reduces the storage requirement from O(k^2) to O(k). Fast Fourier Transform algorithm is utilized to further accelerate the inference by reducing the computational complexity from O(k^2) to O(klogk). The datapath and activation functions are quantized as 16-bit to improve the resource utilization. More importantly, we propose a comprehensive framework called C-LSTM to automatically optimize and implement a wide range of LSTM variants on FPGAs. According to the experimental results, C-LSTM achieves up to 18.8X and 33.5X gains for performance and energy efficiency compared with the state-of-the-art LSTM imple- mentation under the same experimental setup, and the accuracy degradation is very small.


## Installation

## Tutorial


## Publication
If you use [C-LSTM](https://dl.acm.org/doi/10.1145/3174243.3174253) in your design, please cite our [FPGA'18 paper](https://dl.acm.org/doi/10.1145/3174243.3174253):

```
@inproceedings{C-LSTM:FPGA:2018,
  title={C-LSTM: Enabling Efficient LSTM using Structured Compression Techniques on FPGAs},
  author={Shuo Wang and Zhe Li and Caiwen Ding and Bo Yuan and Qinru Qiu and Yanzhi Wang and Yun Liang},
  booktitle={Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA'18)},
  pages={11--20},
  year={2018}
}
```

## Contributing to C-LSTM
1. Use [Pull Request](https://help.github.com/articles/about-pull-requests/).
2. Python [coding style](https://www.python.org/dev/peps/pep-0008/#descriptive-naming-styles).
3. Python [docstring style](https://numpydoc.readthedocs.io/en/latest/format.html#other-points-to-keep-in-mind).
