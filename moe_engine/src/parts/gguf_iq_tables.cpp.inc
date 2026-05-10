// MODULARIZATION: IQ lookup tables split into separate files for better build times
// Original 90KB file split into 7 files by IQ quantization type
// Each file contains one specific IQ grid table for independent compilation

#include "gguf_tables/iq2xs_common.cpp.inc"
#include "gguf_tables/iq2xxs_grid.cpp.inc"
#include "gguf_tables/iq3xxs_grid.cpp.inc"
#include "gguf_tables/iq2xs_grid.cpp.inc"
#include "gguf_tables/iq2s_grid.cpp.inc"
#include "gguf_tables/iq3s_grid.cpp.inc"
#include "gguf_tables/iq1s_grid.cpp.inc"
