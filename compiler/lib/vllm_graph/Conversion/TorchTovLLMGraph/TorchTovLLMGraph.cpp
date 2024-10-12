
#include "vllm_graph/Conversion/TorchTovLLMGraph.hpp"
#include "../PassDetail.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::vllm_graph;
using namespace mlir::torch::Torch;


namespace {

template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <>
LogicalResult ConvertAtenOp<ReluOp>::matchAndRewrite(
    ReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value self = adaptor.getSelf();
    auto selfTy = cast<TensorType>(self.getType());
    if (!selfTy) {
        return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in vllm_graph");
    }

    rewriter.replaceOpWithNewOp<ReluOp>(op, getTypeConverter()->convertType(op.getType()), self);
    return success();

    }
}
