
#include "PassDetail.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include <iostream>

using namespace mlir;
using namespace mlir::vllm_graph;



void replaceFuncDtypes(func::FuncOp &op)
{
    mlir::FunctionType oldFuncType = op.getFunctionType();
    MLIRContext *context = op.getContext();
    mlir::OpBuilder builder(context);

    // Get current argument and result types
    llvm::ArrayRef<mlir::Type> oldArgTypes = oldFuncType.getInputs();
    llvm::ArrayRef<mlir::Type> oldResultTypes = oldFuncType.getResults();

    
    llvm::SmallVector<mlir::Type, 4> newArgTypes;
    //MLIRContext *context = op.getContext();

    auto typeConverter_fn = [context](Type type) {
    Type opType;
    if(auto torchvTensor = cast<mlir::torch::Torch::ValueTensorType>(type)){
        vllm_graph::ValueTensorType vLLMvTensor;
        vLLMvTensor = vLLMvTensor.get(context, 
                        torchvTensor.getOptionalSizes(), 
                        torchvTensor.getOptionalDtype(), 
                        torchvTensor.getOptionalSparsity());
        opType = cast<Type>(vLLMvTensor);
        return opType;
        }
    else
        return opType;
    };
    for (mlir::Type argType : oldArgTypes) {
        if(auto newargType = typeConverter_fn(argType))
            newArgTypes.push_back(newargType);
        else
            newArgTypes.push_back(argType);
    }

    llvm::SmallVector<mlir::Type, 4> newResultTypes;
    for (mlir::Type argType : oldResultTypes) {
        if(auto newargType = typeConverter_fn(argType))
            newResultTypes.push_back(newargType);
        else
            newResultTypes.push_back(argType);  
    }


    // Create the new function type with the updated argument types
    // mlir::FunctionType newFuncType = builder.getFunctionType(newArgTypes, newResultTypes);
    // auto newFuncOp = builder.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), newFuncType);
    mlir::FunctionType newFuncType = builder.getFunctionType(newArgTypes, newResultTypes);
    // auto *entryBlock = &newFuncOp.getBody().front();
    op.setType(newFuncType);
    mlir::Value arg;
    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
        arg = op.getArgument(i);
        arg.setType(newArgTypes[i]);
    // You can now use `arg` for further processing
    }

    mlir::Block &entryBlock = op.getBody().front();
    mlir::Operation &nextOp = entryBlock.front();
    mlir::Location loc = nextOp.getLoc();
     
    builder.setInsertionPointToStart(&entryBlock);
    auto castOp = builder.create<vllm_graph::CastOp>(loc, oldArgTypes[0], arg);
    mlir::Value castOpResult = castOp.getResult();
    nextOp.setOperand(0, castOpResult);

    for(mlir::Operation &currOp : entryBlock)
    {
        if(mlir::isa<func::ReturnOp>(currOp))
        {
            auto returnOp = mlir::cast<func::ReturnOp>(currOp);
            std::cout << __LINE__ << std::endl;
            mlir::Operation *returnOperation = returnOp.getOperation();
            mlir::Value returnValue = returnOperation->getOperand(0);
            builder.setInsertionPoint(returnOp);
            mlir::Location returnloc = returnOp.getLoc();
            auto returnCastOp = builder.create<vllm_graph::CastOp>(returnloc, newResultTypes[0], returnValue);
            returnOperation->setOperand(0, returnCastOp);
        }
    }
}
LogicalResult convertFuncOp(ModuleOp &moduleOp){
    for (auto &op : moduleOp.getBody()->getOperations()) {
        if(auto funcOp = dyn_cast<func::FuncOp>(op))
        {
            replaceFuncDtypes(funcOp);
            //moduleOp.push_back(newFunc);
            return mlir::success();
        }
    }

    return mlir::failure();
}



namespace {
class ConvertGlobalTorchGraph : public ConvertGlobalFunctionPassBase<ConvertGlobalTorchGraph> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<vllm_graph::vLLMGraphIRDialect>();
        registry.insert<func::FuncDialect>();

    }

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        // ConversionTarget target(*context);
        // target.addLegalDialect<vllm_graph::vLLMGraphIRDialect, func::FuncDialect>();

        // TypeConverter typeConverter;
        // typeConverter.addConversion([](Type type) { return type; });

        // target.addIllegalDialect<mlir::torch::Torch::TorchDialect>();

        // RewritePatternSet patterns(context);
        // // target.addIllegalOp<mlir::torch::Torch::AtenReluOp>();                                               
        // // patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenReluOp>>(typeConverter,        
        // //                                                  context);
        // patterns.add<ConvertFuncOp>(typeConverter,context);
        //llvm::outs() << getOperation() << "\n";
    //     for (auto &op : getOperation().getBody()->getOperations()) {
    // // Process each operation here
    // // For example, print the operation
    //         op.print(llvm::outs());
    //     }
        ModuleOp op = getOperation();
        if (failed(convertFuncOp(op)))
            return signalPassFailure();
    }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::vllm_graph::createConvertGlobalFunctionPass(){
    return std::make_unique<ConvertGlobalTorchGraph>();
}



