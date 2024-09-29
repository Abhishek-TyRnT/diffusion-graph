
#ifndef VLLM_GRAPH_DIALECT_VLLM_GRAPH_IR_H
#define VLLM_GRAPH_DIALECT_VLLM_GRAPH_IR_H

#include "mlir/IR/Dialect.h"
#include "vllm_graph/Dialect/IR/vLLMGraphIRDialect.h.inc"

namespace mlir {
namespace vllm_graph {

/// Parse a type registered to this dialect.
Type parseTorchDialectType(AsmParser &parser);

/// Print a type registered to this dialect.
void printTorchDialectType(Type type, AsmPrinter &printer);
} // namespace vllm_graph
} // namespace mlir

#endif