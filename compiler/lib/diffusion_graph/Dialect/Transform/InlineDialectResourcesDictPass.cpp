#include "PassDetail.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "diffusion_graph/Dialect/Transform/Passes.hpp"
#include "mlir/IR/DialectResourceBlobManager.h"
#include <iostream>
using namespace mlir;
using namespace mlir::torch::Torch;

LogicalResult InLineDict(ModuleOp &module){


  // Get tensor data
    module->walk([](mlir::Operation *op) {
        // Print the operation name
        if(isa<ValueTensorLiteralOp>(*op)){
            Builder builder(op);
            ValueTensorLiteralOp literalOp = cast<ValueTensorLiteralOp>(*op);
            ValueTensorType literalType = cast<ValueTensorType>(literalOp.getType());
            Type torchElemType = literalType.getDtype();
            ShapedType srcType;
            ArrayRef<int64_t> shape =  literalType.getSizes();
            auto tensorType = RankedTensorType::get(shape, torchElemType);
            srcType = cast<ShapedType>(tensorType);
            DenseElementsAttr dstElementsAttr;
            if (auto resourceAttr =
                   dyn_cast<DenseResourceElementsAttr>(literalOp.getValue())) {
                
                AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
                if (!blob)
                    literalOp->emitError("could not find resource blob");
                
                ArrayRef<char> ptr = blob->getData();

                // Check that the buffer meets the requirements to get converted to a
                // DenseElementsAttr
                bool detectedSplat = false;
                if (!DenseElementsAttr::isValidRawBuffer(srcType, ptr, detectedSplat))
                    literalOp->emitError("resource is not a valid buffer");
                
                dstElementsAttr =
                    DenseElementsAttr::getFromRawBuffer(resourceAttr.getType(), ptr);
                
                literalOp.setValueAttr(dstElementsAttr);
            }
        }
    });

    return success();

   }



namespace {
class InlineDialectResourcesDict : public diffusion_graph::InlineDialectResourcesDictPassBase<InlineDialectResourcesDict> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect>();
        registry.insert<TorchDialect>();

    }

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        ModuleOp op = getOperation();
        if (failed(InLineDict(op)))
            return signalPassFailure();
    }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::diffusion_graph::createInlineDialectResourcesDictPass(){
    return std::make_unique<InlineDialectResourcesDict>();
}
