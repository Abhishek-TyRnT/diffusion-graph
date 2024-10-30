
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
    /*
    The function replaces the torch.dtypes vllm_graph dtypes. It also adds a
    temporary cast Op, so as to not disturb Ops that have been transformed.

    NOTE: Currently only tested for single return types.
    
    */
    mlir::FunctionType oldFuncType = op.getFunctionType();
    MLIRContext *context = op.getContext();
    mlir::OpBuilder builder(context);

    // Get current argument and result types
    llvm::ArrayRef<mlir::Type> oldArgTypes = oldFuncType.getInputs();
    llvm::ArrayRef<mlir::Type> oldResultTypes = oldFuncType.getResults();

    
    llvm::SmallVector<mlir::Type, 4> newArgTypes;


    //Making a list of dtype conversions from old torch dtypes to new vllm_graph_dtypes.
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

    // Function Type with new dtypes
    mlir::FunctionType newFuncType = builder.getFunctionType(newArgTypes, newResultTypes);

    op.setType(newFuncType);
    mlir::Block &entryBlock = op.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);

    /*Inserting an CastOp so as to not disturb subsequent Ops. 
    Note that in subsequent passes the cast Ops will be eliminated
    */
    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
        mlir::Value arg = op.getArgument(i);
        mlir::Value::user_range opList = arg.getUsers();
        mlir::Location loc = mlir::UnknownLoc::get(context);

        auto castOp = builder.create<vllm_graph::CastOp>(loc, oldArgTypes[0], arg);
        mlir::Value castOpResult = castOp.getResult();
        for (mlir::Operation *user : opList) {
            for(int j = 0; j < user->getNumOperands(); j++)
            {
                // CHecking whether an arg and ops operand are same thing.
                if(user->getOperand(j) == arg)
                    user->setOperand(j, castOpResult);
            }
        }

        arg.setType(newArgTypes[i]);
    }


    //Same logic for return types.
    for(mlir::Operation &currOp : entryBlock)
    {
        if(mlir::isa<func::ReturnOp>(currOp))
        {
            auto returnOp = mlir::cast<func::ReturnOp>(currOp);
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
        ModuleOp op = getOperation();
        if (failed(convertFuncOp(op)))
            return signalPassFailure();
    }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::vllm_graph::createConvertGlobalFunctionPass(){
    return std::make_unique<ConvertGlobalTorchGraph>();
}



