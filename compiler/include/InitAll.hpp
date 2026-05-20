#ifndef DIFFUSION_GRAPH_INITALl_H
#define DIFFUSION_GRAPH_INITALl_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace diffusion_graph{

// Registers all dialects that this project produces and any dependencies.
void registerAllDialects(mlir::DialectRegistry &registry);

// // Registers all necessary dialect extensions for this project
// void registerAllExtensions(mlir::DialectRegistry &registry);

// Registers dialects that may be needed to parse torch-mlir inputs and
// test cases.
void registerOptionalInputDialects(mlir::DialectRegistry &registry);

void registerAllPasses();

} //namespace vllm_graph
} // mlir

#endif // VLLM_GRAPH_INITALl_H
