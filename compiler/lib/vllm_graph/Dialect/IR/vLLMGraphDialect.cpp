
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"

//#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h.inc"

// #include "torch-mlir/Dialect/Torch/IR/TorchDialect.cpp.inc"
// #include "torch-mlir/Dialect/Torch/IR/TorchOps.cpp.inc"


#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vllm_graph;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
#include "vllm_graph/Dialect/IR/vLLMGraphIRDialect.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "torch-mlir/Dialect/Torch/IR/TorchTypes.h.inc"

// #include "torch-mlir/Dialect/Torch/IR/TorchTypes.cpp.inc"


Type vllm_graph::parsevLLMGraphDialectType(AsmParser &parser){
    return Torch::parseTorchDialectType(parser);
}

void vllm_graph::printvLLMGraphDialectType(Type type, AsmPrinter &printer) {
    Torch::printTorchDialectType(type, printer);
}


//===----------------------------------------------------------------------===//
// vLLMGraphIR dialect parseType/printType methods.
//===----------------------------------------------------------------------===//

/// Parse a type registered to this dialect.
Type vLLMGraphIRDialect::parseType(DialectAsmParser &parser) const {
  return parsevLLMGraphDialectType(parser);
}
/// Print a type registered to this dialect.
void vLLMGraphIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printvLLMGraphDialectType(type, printer);
}

void vLLMGraphIRDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "vllm_graph/Dialect/IR/vLLMGraphOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.cpp.inc"
      >();
    
}

Operation *vLLMGraphIRDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
    if (auto integerType = dyn_cast<Torch::IntType>(type))
        return builder.create<Torch::ConstantIntOp>(loc, cast<IntegerAttr>(value));

    if (auto floatType = dyn_cast<Torch::FloatType>(type))
        return builder.create<Torch::ConstantFloatOp>(loc, cast<FloatAttr>(value));

    if (auto numberType = dyn_cast<Torch::NumberType>(type)) {
        if (auto floatValue = dyn_cast<mlir::FloatAttr>(value)) {
        return builder.create<Torch::ConstantNumberOp>(loc, floatValue);
        } else if (auto intValue = dyn_cast<mlir::IntegerAttr>(value)) {
        return builder.create<Torch::ConstantNumberOp>(loc, intValue);
        }
    }

    if (isa<Torch::BoolType>(type)) {
        return builder.create<Torch::ConstantBoolOp>(loc, cast<IntegerAttr>(value));
    }

    if (isa<Torch::NoneType>(type))
        return builder.create<ConstantNoneOp>(loc);

    if (auto stringAttr = dyn_cast<StringAttr>(value))
        return builder.create<ConstantStrOp>(loc, stringAttr);

    if (auto elementsAttr = dyn_cast<ElementsAttr>(value)) {
        // Only !torch.vtensor can be constant folded. !torch.tensor has
        // non-trivial aliasing semantics which prevent deduplicating it.
        assert(isa<ValueTensorType>(type) && "should be a vtensor type!");
        return builder.create<ValueTensorLiteralOp>(loc, elementsAttr);
    }

    return nullptr;
}