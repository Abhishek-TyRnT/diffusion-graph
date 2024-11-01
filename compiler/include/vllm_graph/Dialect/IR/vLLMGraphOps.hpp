
#ifndef VLLM_GRAPH_DIALECT_VLLM_GRAPH_IR_VLLMGRAPHOPS_H
#define VLLM_GRAPH_DIALECT_VLLM_GRAPH_IR_VLLMGRAPHOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "vllm_graph/Dialect/IR/vLLMGraphTrait.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchTraits.h"


#define GET_OP_CLASSES
#include "vllm_graph/Dialect/IR/vLLMGraphOps.h.inc"

namespace mlir {
namespace vllm_graph {
namespace detail {

struct vllm_constant_device_op_binder {
  std::string &bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  vllm_constant_device_op_binder(std::string &bv) : bind_value(bv) {}

  bool match(Operation *op) {
    if (auto constantDevice = dyn_cast<vllm_graph::ConstantDeviceOp>(op)) {
      bind_value = constantDevice.getValue().str();
      return true;
    }
    return false;
  }
};
} //detail
} //vllm_graph
} //mlir

namespace mlir {
namespace vllm_graph {

ParseResult parseDefaultvLLMGraphOp(OpAsmParser &parser, OperationState &result,
                                int numOperands, int numResults);
// Print a generated Torch op in the default format.
void printDefaultvLLMGraphOp(OpAsmPrinter &p, Operation *op, int numOperands,
                         int numResults);

}

}

#endif
