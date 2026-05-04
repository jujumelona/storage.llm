// MODULARIZATION: Codec table split into logical sections for better maintainability
// Original 97KB file split into 4 files by functionality
// Each file contains one specific aspect of model loading and parsing

#include "codec/model_file_io.cpp.inc"
#include "codec/tensor_index_loader.cpp.inc"
#include "codec/gguf_parser.cpp.inc"
#include "codec/juju_parser.cpp.inc"
