#include "diffusion_graph/Conversion/Passes.hpp"


namespace {
#define GEN_PASS_REGISTRATION
#include "diffusion_graph/Conversion/Passes.h.inc"
} // end namespace

void mlir::diffusion_graph::registerConversionPasses() { ::registerPasses(); }