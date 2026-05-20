
#ifndef DIFFUSION_GRAPH_DIALECT_DIFFUSION_GRAPH_IR_H
#define DIFFUSION_GRAPH_DIALECT_DIFFUSION_GRAPH_IR_H

#include "mlir/IR/Dialect.h"
#include "diffusion_graph/Dialect/IR/DiffusionGraphIRDialect.h.inc"

namespace mlir {
namespace diffusion_graph {

/// Parse a type registered to this dialect.
Type parseDiffusionGraphDialectType(AsmParser &parser);

/// Print a type registered to this dialect.
void printDiffusionGraphDialectType(Type type, AsmPrinter &printer);
} // namespace diffusion_graph
} // namespace mlir

#endif