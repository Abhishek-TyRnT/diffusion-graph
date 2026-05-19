//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Attributes.h"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "llvm/ADT/STLExtras.h"


using namespace mlir;
using namespace mlir::diffusion_graph;

//===----------------------------------------------------------------------===//
// isValidSubtype
//===----------------------------------------------------------------------===//
SmallVector<int64_t> makeShapeLLVMCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kUnknownSize);
    if (shape[i] == kUnknownSize)
      updatedShape[i] = kDynamic;
  }
  return updatedShape;
}

bool diffusion_graph::isValidSubtype(Type subtype, Type type) {
  if (subtype == type)
    return true;

  // For a UnionType to be a subtype, all of its contained types must be
  // subtypes.
  if (auto unionType = dyn_cast<UnionType>(subtype)) {
    for (auto containedType : unionType.getContainedTypes()) {
      if (!diffusion_graph::isValidSubtype(containedType, type))
        return false;
    }
    return true;
  }

  if (auto any = dyn_cast<diffusion_graph::AnyType>(type))
    return true;

  if (auto number = dyn_cast<diffusion_graph::NumberType>(type))
    return isa<diffusion_graph::IntType>(subtype) || isa<diffusion_graph::FloatType>(subtype);

  if (auto optional = dyn_cast<diffusion_graph::OptionalType>(type))
    return isValidSubtype(subtype, optional.getContainedType()) ||
           isa<diffusion_graph::NoneType>(subtype);

  if (auto unionType = dyn_cast<UnionType>(type)) {
    for (auto containedType : unionType.getContainedTypes()) {
      if (diffusion_graph::isValidSubtype(subtype, containedType))
        return true;
    }
    return false;
  }

  if (auto tuple = dyn_cast<diffusion_graph::TupleType>(type)) {
    if (!isa<diffusion_graph::TupleType>(subtype))
      return false;
    auto subtypes = cast<diffusion_graph::TupleType>(subtype).getContainedTypes();
    auto types = tuple.getContainedTypes();
    if (subtypes.size() != types.size())
      return false;
    for (auto t : llvm::zip(subtypes, types)) {
      if (!isValidSubtype(std::get<0>(t), std::get<1>(t)))
        return false;
    }
    return true;
  }

  auto subtypeTensorType = dyn_cast<diffusion_graph::BaseTensorType>(subtype);
  auto typeTensorType = dyn_cast<diffusion_graph::BaseTensorType>(type);
  if (subtypeTensorType && typeTensorType) {
    // Check that both tensors have the same `BaseTensorType` subtype.
    // TODO: This is not subtyping according to PEP 483. See description
    // of NonValueTensorType.
    if (isa<diffusion_graph::ValueTensorType>(subtypeTensorType) !=
        isa<diffusion_graph::ValueTensorType>(typeTensorType))
      return false;

    // `type` must not have more static information than `subtype`, and `type`
    // must not disagree with `subtype`.
    if (typeTensorType.hasDtype() &&
        (!subtypeTensorType.hasDtype() ||
         typeTensorType.getDtype() != subtypeTensorType.getDtype())) {
      return false;
    }

    // `type` must not have more static shape information than `subtype`.
    auto isSubsizes = [](diffusion_graph::BaseTensorType type, diffusion_graph::BaseTensorType subtype) -> bool {
      auto typeSizes = type.getSizes();
      auto subtypeSizes = subtype.getSizes();
      if (typeSizes.size() != subtypeSizes.size()) {
        return false;
      }
      for (auto t : llvm::zip(typeSizes, subtypeSizes)) {
        if (std::get<0>(t) != kUnknownSize &&
            std::get<0>(t) != std::get<1>(t)) {
          return false;
        }
      }
      return true;
    };

    if (typeTensorType.hasSizes() &&
        (!subtypeTensorType.hasSizes() ||
         !isSubsizes(typeTensorType, subtypeTensorType))) {
      return false;
    }

    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Helpers for TupleType and UnionType
//===----------------------------------------------------------------------===//

// Parse the `<T1, T2, T3>` of a type such as `!torch.tuple<T1, T2, T3>`.
static std::optional<SmallVector<Type>>
parseMultipleContainedTypes(AsmParser &parser) {
  if (parser.parseLess())
    return std::nullopt;

  SmallVector<Type> containedTypes;
  if (!parser.parseOptionalGreater())
    return containedTypes;
  do {
    Type containedType = parseDiffusionGraphDialectType(parser);
    if (!containedType)
      return std::nullopt;
    containedTypes.push_back(containedType);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return std::nullopt;
  return containedTypes;
}

static void printMultipleContainedTypes(AsmPrinter &printer,
                                        ArrayRef<Type> containedTypes) {
  printer << "<";
  llvm::interleaveComma(containedTypes, printer, [&](Type type) {
    printDiffusionGraphDialectType(type, printer);
  });
  printer << ">";
}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

Type diffusion_graph::TupleType::parse(AsmParser &parser) {
  if (auto containedTypes = parseMultipleContainedTypes(parser))
    return TupleType::get(parser.getContext(), *containedTypes);
  return Type();
}

void diffusion_graph::TupleType::print(AsmPrinter &printer) const {
  printMultipleContainedTypes(printer, getContainedTypes());
}

//===----------------------------------------------------------------------===//
// UnionType
//===----------------------------------------------------------------------===//

Type diffusion_graph::UnionType::parse(AsmParser &parser) {
  if (auto containedTypes = parseMultipleContainedTypes(parser))
    return UnionType::get(parser.getContext(), *containedTypes);
  return Type();
}

void diffusion_graph::UnionType::print(AsmPrinter &printer) const {
  printMultipleContainedTypes(printer, getContainedTypes());
}

//===----------------------------------------------------------------------===//
// BaseTensorType
//===----------------------------------------------------------------------===//

static bool isValidTorchDtype(Type dtype) {
  // For complex types, get the underlying element type
  if (isa<ComplexType>(dtype)) {
    dtype = cast<ComplexType>(dtype).getElementType();
  }
  // Torch quantized types.
  if (isa<diffusion_graph::QInt8Type, diffusion_graph::QUInt8Type, diffusion_graph::QInt16Type,
          diffusion_graph::QInt32Type>(dtype))
    return true;
  // Builtin floating point types.
  if (isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(dtype))
    return true;
  if (isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
          Float8E4M3FNUZType, Float8E4M3B11FNUZType>(dtype))
    return true;

  if (isa<diffusion_graph::StringType>(dtype))
    return true;
  // Builtin integer types.
  if (IntegerType type = dyn_cast<IntegerType>(dtype)) {
    if (type.isSignless() && type.getWidth() == 1)
      return true;
    if (type.isSigned()) {
      for (unsigned width : {4, 8, 16, 32, 64}) {
        if (type.getWidth() == width)
          return true;
      }
    }
    if (type.isUnsigned()) {
      for (unsigned width : {4, 8, 16, 32, 64}) {
        if (type.getWidth() == width)
          return true;
      }
    }
  }
  return false;
}

bool diffusion_graph::BaseTensorType::hasSameSizesAndDtype(diffusion_graph::BaseTensorType other) const {
  return getOptionalSizes() == other.getOptionalSizes() &&
         getOptionalDtype() == other.getOptionalDtype();
}

Type diffusion_graph::BaseTensorType::getWithSizesAndDtypeFrom(diffusion_graph::BaseTensorType other) const {
  return getWithSizesAndDtype(other.getOptionalSizes(),
                              other.getOptionalDtype());
}

Type diffusion_graph::BaseTensorType::getWithSizesAndDtype(
    std::optional<ArrayRef<int64_t>> optionalSizes, Type optionalDtype) const {
  if (mlir::isa<diffusion_graph::NonValueTensorType>(*this))
    return diffusion_graph::NonValueTensorType::get(getContext(), optionalSizes, optionalDtype);
  if (mlir::isa<diffusion_graph::ValueTensorType>(*this))
    return diffusion_graph::ValueTensorType::get(getContext(), optionalSizes, optionalDtype);
  llvm_unreachable("not a BaseTensorType!");
}

Type diffusion_graph::BaseTensorType::getWithSizesAndDtypeAndSparsity(
    std::optional<ArrayRef<int64_t>> optionalSizes, Type optionalDtype,
    Attribute optionalSparsity) const {
  if (mlir::isa<diffusion_graph::NonValueTensorType>(*this))
    return diffusion_graph::NonValueTensorType::get(getContext(), optionalSizes, optionalDtype,
                                   optionalSparsity);
  if (mlir::isa<diffusion_graph::ValueTensorType>(*this))
    return diffusion_graph::ValueTensorType::get(getContext(), optionalSizes, optionalDtype,
                                optionalSparsity);
  llvm_unreachable("not a BaseTensorType!");
}

diffusion_graph::ValueTensorType diffusion_graph::BaseTensorType::getWithValueSemantics() const {
  if (auto tensor = mlir::dyn_cast<diffusion_graph::NonValueTensorType>(*this))
    return tensor.getWithValueSemantics();
  if (auto tensor = mlir::dyn_cast<diffusion_graph::ValueTensorType>(*this))
    return tensor;
  llvm_unreachable("not a BaseTensorType!");
}

static LogicalResult
verifyTensorType(function_ref<InFlightDiagnostic()> emitError,
                 std::optional<ArrayRef<int64_t>> optionalSizes,
                 Type optionalDtype, Attribute optionalSparsity) {
  if (optionalDtype && !isValidTorchDtype(optionalDtype)) {
    emitError() << "invalid dtype " << optionalDtype
                << " for !torch.tensor type";
    return failure();
  }
  if (optionalSizes.has_value()) {
    for (int64_t size : optionalSizes.value()) {
      if (size < 0 && size != kUnknownSize) {
        emitError() << "invalid size " << size << " for !torch.tensor type";
        return failure();
      }
    }
  }
  // Verify sparsity encoding against a known type and shape using the encoding
  // verification interface. Any implementation emits a diagnostic on failure.
  // Also verify sparsity encoding is truly a sparse encoding attrbute.
  // if (optionalSparsity) {
  //   if (optionalDtype && optionalSizes.has_value()) {
  //     if (auto venc = llvm::dyn_cast_or_null<VerifiableTensorEncoding>(
  //             optionalSparsity)) {
  //       if (failed(venc.verifyEncoding(optionalSizes.value(), optionalDtype,
  //                                      emitError))) {
  //         return failure();
  //       }
  //     }
  //   }
  //   if (!isa<sparse_tensor::SparseTensorEncodingAttr>(optionalSparsity)) {
  //     emitError() << "invalid sparsity encoding attribute";
  //     return failure();
  //   }
  // }
  return success();
}

Type parseDiffusionTensorType(MLIRContext *context, AsmParser &parser,
                     GetTensorTypeFn getTensorType) {
  llvm::SMLoc startLoc = parser.getCurrentLocation();
  if (parser.parseOptionalLess())
    return getTensorType(context,
                         /*optionalSizes=*/std::nullopt,
                         /*optionalDtype=*/Type(),
                         /*optionalSparsity=*/Attribute());
  bool hasSizes;
  SmallVector<int64_t> sizes;
  if (succeeded(parser.parseOptionalStar())) {
    // Unranked.
    hasSizes = false;
  } else {
    // Parse list of sizes.
    hasSizes = true;
    if (parser.parseLSquare())
      return Type();
    for (bool first = true;; first = false) {
      if (!first) {
        if (failed(parser.parseOptionalComma())) {
          break;
        }
      }
      if (succeeded(parser.parseOptionalQuestion())) {
        sizes.push_back(-1);
        continue;
      }
      int64_t size;
      auto optionalInt = parser.parseOptionalInteger(size);
      if (optionalInt.has_value()) {
        if (failed(*optionalInt))
          return Type();
        sizes.push_back(size);
        continue;
      }
      break;
    }
    if (parser.parseRSquare()) {
      return Type();
    }
  }
  if (parser.parseComma())
    return Type();
  Type optionalDtype;
  if (succeeded(parser.parseOptionalKeyword("unk"))) {
    // Unknown dtype.
  } else {
    // Known dtype.
    if (parser.parseType(optionalDtype))
      return Type();
  }
  Attribute optionalSparsity;
  if (succeeded(parser.parseOptionalComma())) {
    // Explicit encoding.
    if (parser.parseAttribute(optionalSparsity))
      return Type();
  }
  if (parser.parseGreater())
    return Type();
  std::optional<ArrayRef<int64_t>> optionalSizes;
  if (hasSizes)
    optionalSizes.emplace(sizes);

  if (failed(verifyTensorType([&]() { return parser.emitError(startLoc); },
                              optionalSizes, optionalDtype, optionalSparsity)))
    return Type();

  return getTensorType(context, optionalSizes, optionalDtype, optionalSparsity);
}

static void printDiffusionTensorType(AsmPrinter &printer,
                            std::optional<ArrayRef<int64_t>> optionalSizes,
                            Type optionalDtype, Attribute optionalSparsity) {
  if (!optionalSizes && !optionalDtype)
    return;
  printer << "<";
  if (optionalSizes) {
    printer << "[";
    for (auto it : llvm::enumerate(*optionalSizes)) {
      if (it.index() > 0)
        printer << ",";
      if (it.value() < 0)
        printer << "?";
      else
        printer << it.value();
    }
    printer << "]";
  } else {
    printer << "*";
  }
  printer << ",";
  if (optionalDtype)
    printer.printType(optionalDtype);
  else
    printer << "unk";
  if (optionalSparsity) {
    printer << ",";
    printer.printAttribute(optionalSparsity);
  }
  printer << ">";
}

//===----------------------------------------------------------------------===//
// NonValueTensorType
//===----------------------------------------------------------------------===//

diffusion_graph::ValueTensorType diffusion_graph::NonValueTensorType::getWithValueSemantics() const {
  return diffusion_graph::ValueTensorType::get(getContext(), getOptionalSizes(),
                              getOptionalDtype());
}

diffusion_graph::NonValueTensorType
diffusion_graph::NonValueTensorType::getWithLeastStaticInformation(MLIRContext *context) {
  return NonValueTensorType::get(context,
                                 /*optionalSizes=*/std::nullopt,
                                 /*optionalDtype=*/Type());
}

LogicalResult
diffusion_graph::NonValueTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                           std::optional<ArrayRef<int64_t>> optionalSizes,
                           Type optionalDtype, Attribute optionalSparsity) {
  return verifyTensorType(emitError, optionalSizes, optionalDtype,
                          optionalSparsity);
}

Type diffusion_graph::NonValueTensorType::parse(AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  return parseDiffusionTensorType(
      context, parser,
      [](MLIRContext *context, std::optional<ArrayRef<int64_t>> optionalSizes,
         Type optionalType, Attribute optionalSparsity) {
        return diffusion_graph::NonValueTensorType::get(context, optionalSizes, optionalType,
                                       optionalSparsity);
      });
}

void diffusion_graph::NonValueTensorType::print(AsmPrinter &printer) const {
  printDiffusionTensorType(printer, getOptionalSizes(), getOptionalDtype(),
                  getOptionalSparsity());
}

//===----------------------------------------------------------------------===//
// ValueTensorType
//===----------------------------------------------------------------------===//

diffusion_graph::NonValueTensorType diffusion_graph::ValueTensorType::getWithoutValueSemantics() const {
  return diffusion_graph::NonValueTensorType::get(getContext(), getOptionalSizes(),
                                 getOptionalDtype());
}

diffusion_graph::ValueTensorType
diffusion_graph::ValueTensorType::getWithLeastStaticInformation(MLIRContext *context) {
  return diffusion_graph::ValueTensorType::get(context,
                              /*optionalSizes=*/std::nullopt,
                              /*optionalDtype=*/Type());
}

static Type convertDtypeToBuiltinElementType(MLIRContext *context, Type dtype) {
  if (isa<mlir::FloatType, IntegerType, mlir::ComplexType>(dtype)) {
    return dtype;
  }

  if (isa<diffusion_graph::QUInt8Type>(dtype))
    return IntegerType::get(context, 8, IntegerType::Signless);

  if (isa<diffusion_graph::QInt8Type>(dtype))
    return IntegerType::get(context, 8, IntegerType::Signless);

  if (isa<diffusion_graph::QInt16Type>(dtype))
    return IntegerType::get(context, 16, IntegerType::Signless);

  if (isa<diffusion_graph::QInt32Type>(dtype))
    return IntegerType::get(context, 32, IntegerType::Signless);

  emitError(UnknownLoc::get(context))
      << "unimplemented: conversion of dtype " << dtype
      << " to builtin tensor element type";
  return nullptr;
}

TensorType diffusion_graph::ValueTensorType::toBuiltinTensor() const {
  if (!hasDtype())
    return nullptr;
  Type elementType = convertDtypeToBuiltinElementType(getContext(), getDtype());
  if (!elementType)
    return nullptr;
  if (!hasSizes())
    return UnrankedTensorType::get(elementType);
  return RankedTensorType::get(makeShapeLLVMCompatible(getSizes()), elementType,
                               getOptionalSparsity());
}

LogicalResult
diffusion_graph::ValueTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                        std::optional<ArrayRef<int64_t>> optionalSizes,
                        Type optionalDtype, Attribute optionalSparsity) {
  return verifyTensorType(emitError, optionalSizes, optionalDtype,
                          optionalSparsity);
}

Type ValueTensorType::parse(AsmParser &parser) {
  MLIRContext *context = parser.getContext();
  return parseDiffusionTensorType(
      context, parser,
      [](MLIRContext *context, std::optional<ArrayRef<int64_t>> optionalSizes,
         Type optionalType, Attribute optionalSparsity) {
        return ValueTensorType::get(context, optionalSizes, optionalType,
                                    optionalSparsity);
      });
}

void ValueTensorType::print(AsmPrinter &printer) const {
  printDiffusionTensorType(printer, getOptionalSizes(), getOptionalDtype(),
                  getOptionalSparsity());
}

Type meetTensorTypes(diffusion_graph::BaseTensorType lhs, diffusion_graph::BaseTensorType rhs) {
  assert(((isa<diffusion_graph::ValueTensorType>(lhs) && isa<diffusion_graph::ValueTensorType>(rhs)) ||
          (isa<diffusion_graph::NonValueTensorType>(lhs) && isa<diffusion_graph::NonValueTensorType>(rhs))) &&
         "expected lhs and rhs to have same sense of value semantics");

  // First, calculate the dtype.

  // If the dtypes are contradictory, return null.
  if (lhs.hasDtype() && rhs.hasDtype() && lhs.getDtype() != rhs.getDtype())
    return nullptr;
  Type dtype;
  // If we have a dtype, use it. If not, then the dtype Type remains in its
  // default null state, which the constructor of ValueTensorType treats as
  // "unknown".
  if (lhs.hasDtype() || rhs.hasDtype()) {
    dtype = lhs.hasDtype() ? lhs.getDtype() : rhs.getDtype();
  }

  // Then, calculate the sizes and return the new Type.

  // If neither has sizes, we have nothing left to do.
  if (!lhs.hasSizes() && !rhs.hasSizes()) {
    return diffusion_graph::ValueTensorType::get(lhs.getContext(),
                                /*optionalSizes=*/std::nullopt, dtype);
  }

  // If the number of sizes is different, the two types are contradictory.
  if (lhs.hasSizes() && rhs.hasSizes() &&
      lhs.getSizes().size() != rhs.getSizes().size()) {
    return nullptr;
  }

  // Either lhs or rhs has sizes. If either one doesn't have sizes, we can
  // replace it with the other one's sizes, since the meet logic below is
  // idempotent.
  ArrayRef<int64_t> lhsSizes = lhs.hasSizes() ? lhs.getSizes() : rhs.getSizes();
  ArrayRef<int64_t> rhsSizes = rhs.hasSizes() ? rhs.getSizes() : lhs.getSizes();
  // Meet the sizes.
  SmallVector<int64_t> newSizes;
  for (int i = 0, e = lhsSizes.size(); i < e; i++) {
    if (lhsSizes[i] == rhsSizes[i]) {
      newSizes.push_back(lhsSizes[i]);
    } else if (lhsSizes[i] == kUnknownSize) {
      newSizes.push_back(rhsSizes[i]);
    } else if (rhsSizes[i] == kUnknownSize) {
      newSizes.push_back(lhsSizes[i]);
    } else {
      // The two sizes are contradictory.
      return nullptr;
    }
  }

  return lhs.getWithSizesAndDtype(ArrayRef(newSizes), dtype);
}

////===----------------------------------------------------------------------===//
//// DictType
////===----------------------------------------------------------------------===//

// TODO: These are not DRY in that the two type predicates AnyTorchDictKeyType
// and AnyTorchType generate the exact same code (in TorchOps.cpp.inc).
// Unfortunately the generated implementations aren't visible/exposed ("static"
// linkage) and the predicates themselves can't be added/used in the
// specification of the parameters of the Torch_DictType.
static bool isAnyTorchDictKeyType(Type type) {
  return isa<diffusion_graph::AnyType>(type) || isa<diffusion_graph::IntType>(type) ||
         isa<diffusion_graph::BoolType>(type) || isa<diffusion_graph::FloatType>(type) ||
         isa<diffusion_graph::StringType>(type) || isa<diffusion_graph::BaseTensorType>(type);
}

static bool isAnyTorchType(Type type) {
  return isValidSubtype(type, diffusion_graph::NumberType::get(type.getContext())) ||
         isa<diffusion_graph::BaseTensorType>(type) || isa<diffusion_graph::AnyType>(type) ||
         isa<diffusion_graph::BoolType>(type) || isa<diffusion_graph::DictType>(type) ||
         isa<diffusion_graph::DeviceType>(type) || isa<diffusion_graph::GeneratorType>(type) ||
         isa<diffusion_graph::ListType>(type) || isa<diffusion_graph::LinearParamsType>(type) ||
         isa<diffusion_graph::NumberType>(type) || isa<diffusion_graph::NnModuleType>(type) ||
         isa<diffusion_graph::NoneType>(type) || isa<diffusion_graph::OptionalType>(type) ||
         isa<diffusion_graph::StringType>(type) || isa<diffusion_graph::TupleType>(type) ||
         isa<diffusion_graph::UnionType>(type);
}

LogicalResult
DictType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                 Type keyType, Type valueType) {
  if (!isAnyTorchDictKeyType(keyType)) {
    emitError() << "invalid " << keyType << " for !torch.dict key type";
    return failure();
  }
  if (!isAnyTorchType(valueType)) {
    emitError() << "invalid " << valueType << " for !torch.dict value type";
    return failure();
  }
  return success();
}
