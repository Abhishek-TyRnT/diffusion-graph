
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"

// #include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
// #include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
//#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

// #include "torch-mlir/Dialect/Torch/IR/TorchDialect.cpp.inc"
// #include "torch-mlir/Dialect/Torch/IR/TorchOps.cpp.inc"


#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::vllm_graph;
// using namespace mlir::torch;
// using namespace mlir::torch::Torch;
#include "vllm_graph/Dialect/IR/vLLMGraphIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.cpp.inc"


Type vllm_graph::parsevLLMGraphDialectType(AsmParser &parser){
    SMLoc typeLoc = parser.getCurrentLocation();
    StringRef mnemonic;
    Type genType;
    auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
    if(parseResult.has_value())
        return genType;
    parser.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `"
                                << vLLMGraphIRDialect::getDialectNamespace() << "`";
    return {};
}

void vllm_graph::printvLLMGraphDialectType(Type type, AsmPrinter &printer) {
    if (succeeded(generatedTypePrinter(type, printer)))
        return;
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
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.cpp.inc"
      >();
    
}

Operation *vLLMGraphIRDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
    if (auto integerType = dyn_cast<vllm_graph::IntType>(type))
        return builder.create<arith::ConstantIntOp>(loc, cast<IntegerAttr>(value).getValue().getSExtValue(), builder.getIntegerType(32));

    if (auto floatType = dyn_cast<vllm_graph::FloatType>(type))
        return builder.create<arith::ConstantFloatOp>(loc, cast<FloatAttr>(value).getValue(), builder.getF32Type());


    // TODO: Add number type
    // if (auto numberType = dyn_cast<vllm_graph::NumberType>(type)) {
    //     if (auto floatValue = dyn_cast<mlir::FloatAttr>(value)) {
    //     return builder.create<vllm_graph::ConstantNumberOp>(loc, floatValue);
    //     } else if (auto intValue = dyn_cast<mlir::IntegerAttr>(value)) {
    //     return builder.create<vllm_graph::ConstantNumberOp>(loc, intValue);
    //     }
    // }

    if (isa<vllm_graph::BoolType>(type)) {
        return builder.create<arith::ConstantIntOp>(loc, cast<IntegerAttr>(value).getValue().getZExtValue(), builder.getI1Type());
    }

    // if (isa<vllm_graph::NoneType>(type))
    //     return builder.create<arith::ConstantOp>(loc, mlir::NoneType::get(builder.getContext()));
;

    // if (auto stringAttr = dyn_cast<StringAttr>(value))
    //     return builder.create<ConstantStrOp>(loc, stringAttr);

    // if (auto elementsAttr = dyn_cast<vllm_graph::ElementsAttr>(value)) {
    //     // Only !torch.vtensor can be constant folded. !torch.tensor has
    //     // non-trivial aliasing semantics which prevent deduplicating it.
    //     assert(isa<vllm_graph::ValueTensorType>(type) && "should be a vtensor type!");
    //     return builder.create<vllm_graph::ValueTensorLiteralOp>(loc, elementsAttr);
    // }

    return nullptr;
}

void OptionalType::print(AsmPrinter &printer) const {
  printer << "<";
  // Print the contained type without the `!torch.` prefix.
  printvLLMGraphDialectType(getImpl()->containedType, printer);
  printer << ">";
}

void ListType::print(AsmPrinter &printer) const {
  printer << "<";
  // Print the contained type without the `!torch.` prefix.
  printvLLMGraphDialectType(getImpl()->containedType, printer);
  printer << ">";
}

Type OptionalType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();

  // Parse the contained type, but forward directly to our internal parsing
  // of `torch` dialect types, so that we can parse nested types without
  // the `!torch.` prefix.
  Type containedType = parsevLLMGraphDialectType(odsParser);
  if (!containedType)
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), containedType);
}

Type ListType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();

  // Parse the contained type, but forward directly to our internal parsing
  // of `torch` dialect types, so that we can parse nested types without
  // the `!torch.` prefix.
  Type containedType = parsevLLMGraphDialectType(odsParser);
  if (!containedType)
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), containedType);
}

//===----------------------------------------------------------------------===//
// DictType
//===----------------------------------------------------------------------===//

void DictType::print(AsmPrinter &printer) const {
  printer << "<";
  printvLLMGraphDialectType(getImpl()->keyType, printer);
  printer << ", ";
  printvLLMGraphDialectType(getImpl()->valueType, printer);
  printer << ">";
}

Type DictType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();
  Type keyType = parsevLLMGraphDialectType(odsParser);
  if (!keyType)
    return Type();
  if (odsParser.parseComma())
    return Type();
  Type valueType = parsevLLMGraphDialectType(odsParser);
  if (!valueType)
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), keyType, valueType);
}

//===----------------------------------------------------------------------===//
// NnModuleType
//===----------------------------------------------------------------------===//

void NnModuleType::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getImpl()->className, printer.getStream());
  printer << "\">";
}

Type NnModuleType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();
  std::string className;
  if (odsParser.parseOptionalString(&className))
    return Type();
  if (odsParser.parseGreater())
    return Type();
  return get(odsParser.getContext(), className);
}
