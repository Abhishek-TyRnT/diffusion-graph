#include "vllm_graph/Utils/Utils.hpp"

using namespace mlir;
using namespace mlir::vllm_graph;

Type mlir::vllm_graph::convertTorchvTypeTovLLMvType(Type type, MLIRContext *context){

    auto torchvTensor = cast<mlir::torch::Torch::ValueTensorType>(type);
    if(torchvTensor){
        Type opType;
        vllm_graph::ValueTensorType vLLMvTensor;
        vLLMvTensor = vLLMvTensor.get(context, 
                        torchvTensor.getOptionalSizes(), 
                        torchvTensor.getOptionalDtype(), 
                        torchvTensor.getOptionalSparsity());
        opType = cast<Type>(vLLMvTensor);
        return opType;
    }

    else
        return type;
}

RankedTensorType mlir::vllm_graph::convertTorchvTypeToTensorType(Type type){
    
    auto TorchTensor = cast<mlir::torch::Torch::ValueTensorType>(type);
    Type elemType = TorchTensor.getOptionalDtype();
    SmallVector<int64_t> shape = cast<SmallVector<int64_t>>(TorchTensor.getOptionalSizes());
    
    RankedTensorType tensor = RankedTensorType::get(shape, elemType);
    return tensor;
}

Type mlir::vllm_graph::convertvLLMContainedType(Type type, 
                        ConversionPatternRewriter &rewriter, 
                        MLIRContext *context){
    auto TorchList = cast<torch::Torch::ListType>(type);

    Type containedResultType;
    if(isa<torch::Torch::IntType>(TorchList.getContainedType()))
        containedResultType = rewriter.getIntegerType(32);
    else if(isa<torch::Torch::FloatType>(TorchList.getContainedType()))
        containedResultType = rewriter.getF32Type();
    else if(isa<torch::Torch::BoolType>(TorchList.getContainedType()))
        containedResultType = rewriter.getI1Type();
    else if(isa<torch::Torch::ValueTensorType>(TorchList.getContainedType())){
        containedResultType = vllm_graph::convertTorchvTypeTovLLMvType(TorchList.getContainedType(), context);
    }
    else
        assert(false && "Type for the list not added");

    return containedResultType;
}