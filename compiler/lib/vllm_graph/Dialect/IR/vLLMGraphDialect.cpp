
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



#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::diffusion_graph;

#include "vllm_graph/Dialect/IR/DiffusionGraphIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.cpp.inc"


Type diffusion_graph::parseDiffusionGraphDialectType(AsmParser &parser){
    SMLoc typeLoc = parser.getCurrentLocation();
    StringRef mnemonic;
    Type genType;
    auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
    if(parseResult.has_value())
        return genType;
    parser.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `"
                                << DiffusionGraphIRDialect::getDialectNamespace() << "`";
    return {};
}

void diffusion_graph::printDiffusionGraphDialectType(Type type, AsmPrinter &printer) {
    if (succeeded(generatedTypePrinter(type, printer)))
        return;
}


//===----------------------------------------------------------------------===//
// vLLMGraphIR dialect parseType/printType methods.
//===----------------------------------------------------------------------===//

/// Parse a type registered to this dialect.
Type DiffusionGraphIRDialect::parseType(DialectAsmParser &parser) const {
  return parseDiffusionGraphDialectType(parser);
}
/// Print a type registered to this dialect.
void DiffusionGraphIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printDiffusionGraphDialectType(type, printer);
}

void DiffusionGraphIRDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "vllm_graph/Dialect/IR/vLLMGraphOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.cpp.inc"
      >();
    
}

Operation *DiffusionGraphIRDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
    // if (auto integerType = dyn_cast<diffusion_graph::IntType>(type))
    //     return builder.create<arith::ConstantIntOp>(loc, cast<IntegerAttr>(value).getValue().getSExtValue(), builder.getIntegerType(32));

    if (auto floatType = dyn_cast<diffusion_graph::FloatType>(type))
        return builder.create<arith::ConstantFloatOp>(loc, cast<FloatAttr>(value).getValue(), builder.getF32Type());

    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      if (isa<IntegerType>(type)) {
        // Use your dialect's constant op instead of arith.constant
        return builder.create<arith::ConstantIntOp>(loc, intAttr.getInt(), type);
      }
   }


    // TODO: Add number type
    // if (auto numberType = dyn_cast<diffusion_graph::NumberType>(type)) {
    //     if (auto floatValue = dyn_cast<mlir::FloatAttr>(value)) {
    //     return builder.create<diffusion_graph::ConstantNumberOp>(loc, floatValue);
    //     } else if (auto intValue = dyn_cast<mlir::IntegerAttr>(value)) {
    //     return builder.create<diffusion_graph::ConstantNumberOp>(loc, intValue);
    //     }
    // }

    if (isa<diffusion_graph::BoolType>(type)) {
        return builder.create<arith::ConstantIntOp>(loc, cast<IntegerAttr>(value).getValue().getZExtValue(), builder.getI1Type());
    }

    // if (isa<diffusion_graph::NoneType>(type))
    //     return builder.create<arith::ConstantOp>(loc, mlir::NoneType::get(builder.getContext()));
;

    // if (auto stringAttr = dyn_cast<StringAttr>(value))
    //     return builder.create<ConstantStrOp>(loc, stringAttr);

    // if (auto elementsAttr = dyn_cast<diffusion_graph::ElementsAttr>(value)) {
    //     // Only !torch.vtensor can be constant folded. !torch.tensor has
    //     // non-trivial aliasing semantics which prevent deduplicating it.
    //     assert(isa<diffusion_graph::ValueTensorType>(type) && "should be a vtensor type!");
    //     return builder.create<diffusion_graph::ValueTensorLiteralOp>(loc, elementsAttr);
    // }

    return nullptr;
}

void OptionalType::print(AsmPrinter &printer) const {
  printer << "<";
  // Print the contained type without the `!torch.` prefix.
  printDiffusionGraphDialectType(getImpl()->containedType, printer);
  printer << ">";
}

void ListType::print(AsmPrinter &printer) const {
  printer << "<";
  // Print the contained type without the `!torch.` prefix.
  printer.printType(getImpl()->containedType);
  printer << ">";
}

Type OptionalType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();

  // Parse the contained type, but forward directly to our internal parsing
  // of `torch` dialect types, so that we can parse nested types without
  // the `!torch.` prefix.
  Type containedType = parseDiffusionGraphDialectType(odsParser);
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
  Type containedType = parseDiffusionGraphDialectType(odsParser);
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
  printDiffusionGraphDialectType(getImpl()->keyType, printer);
  printer << ", ";
  printDiffusionGraphDialectType(getImpl()->valueType, printer);
  printer << ">";
}

Type DictType::parse(AsmParser &odsParser) {
  if (odsParser.parseLess())
    return Type();
  Type keyType = parseDiffusionGraphDialectType(odsParser);
  if (!keyType)
    return Type();
  if (odsParser.parseComma())
    return Type();
  Type valueType = parseDiffusionGraphDialectType(odsParser);
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
