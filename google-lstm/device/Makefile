XILINX_SDACCEL := /opt/Xilinx/SDx/2017.1
XILINX_OPENCL := $(XILINX_SDACCEL)
#DSA := xilinx:adm-pcie-7v3:1ddr:3.0 
DSA := xilinx:adm-pcie-ku3:2ddr:3.3
XOCC := $(XILINX_SDACCEL)/bin/xocc
CLFLAGS := -s --xdevice $(DSA)

all : xilinx_sda_lstm.xclbin \
      xilinx_sda_lstm.xo \

.PHONY : all

xilinx_sda_lstm.xclbin: ./lstm_fpga.cpp
	$(XOCC) $(CLFLAGS) -k lstm --kernel_frequency 200 --optimize 3 -r system $< -o $@ ;\

xilinx_sda_lstm.xo: ./lstm_fpga.cpp
	$(XOCC) $(CLFLAGS) -c -k lstm -r estimate --kernel_frequency 200 $< -o $@ ;\

xilinx_sda_compute_gate.xo: ./compute_gate.cpp
	$(XOCC) $(CLFLAGS) -c -k compute_gate -r estimate --kernel_frequency 200 $< -o $@ ;\

xilinx_sda_compute_yt.xo: ./compute_yt.cpp
	$(XOCC) $(CLFLAGS) -c -k compute_yt -r estimate --kernel_frequency 200 $< -o $@ ;\

xilinx_sda_compute_vector.xo: ./compute_vector.cpp
	$(XOCC) $(CLFLAGS) -c -k compute_vector -r estimate --kernel_frequency 200 $< -o $@ ;\
