// MODULARIZATION: CPU tensor dot kernels split by quantization type
// Original 56KB file split into 4 files for parallel compilation

#include "tensor_kernels/dot_fp_kernels.cpp.inc"
#include "tensor_kernels/dot_q8q4q5_kernels.cpp.inc"
#include "tensor_kernels/dot_kquant_kernels.cpp.inc"
#include "tensor_kernels/dot_iq_kernels.cpp.inc"
