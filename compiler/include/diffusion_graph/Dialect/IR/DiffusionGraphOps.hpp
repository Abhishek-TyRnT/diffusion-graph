
#ifndef DIFFUSION_GRAPH_DIALECT_DIFFUSION_GRAPH_IR_DIFFUSIONGRAPHOPS_H
#define DIFFUSION_GRAPH_DIALECT_DIFFUSION_GRAPH_IR_DIFFUSIONGRAPHOPS_H

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

#include "diffusion_graph/Dialect/IR/DiffusionGraphTrait.hpp"
#include "diffusion_graph/Dialect/IR/DiffusionGraphOps.hpp"
#include "diffusion_graph/Dialect/IR/DiffusionGraphTypes.hpp"

#define GET_OP_CLASSES
#include "diffusion_graph/Dialect/IR/DiffusionGraphOps.h.inc"

namespace mlir {
namespace diffusion_graph {
namespace detail {

struct vllm_constant_device_op_binder {
  std::string &bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  vllm_constant_device_op_binder(std::string &bv) : bind_value(bv) {}

  bool match(Operation *op) {
    if (auto constantDevice = dyn_cast<diffusion_graph::ConstantDeviceOp>(op)) {
      bind_value = constantDevice.getValue().str();
      return true;
    }
    return false;
  }
};
} //detail
} //diffusion_graph
} //mlir

namespace mlir {
namespace diffusion_graph {

ParseResult parseDefaultDiffusionGraphOp(OpAsmParser &parser, OperationState &result,
                                int numOperands, int numResults);
// Print a generated Diffusion Graph op in the default format.
void printDefaultDiffusionGraphOp(OpAsmPrinter &p, Operation *op, int numOperands,
                         int numResults);

}

}

#endif
