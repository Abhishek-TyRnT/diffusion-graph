#include "vllm_graph/Conversion/Passes.h"


namespace {
#define GEN_PASS_REGISTRATION
#include "vllm_graph/Conversion/Passes.h.inc"
} // end namespace

void mlir::vllm_graph::registerConversionPasses() { ::registerPasses(); }