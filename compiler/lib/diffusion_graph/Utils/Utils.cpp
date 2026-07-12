#include "diffusion_graph/Utils/Utils.hpp"

using namespace mlir;
using namespace mlir::diffusion_graph;

Type mlir::diffusion_graph::convertTorchvTypeToDGvType(Type type, MLIRContext *context){

    auto torchvTensor = cast<mlir::torch::Torch::ValueTensorType>(type);
    if(torchvTensor){
        Type opType;
        Type elemType = torchvTensor.getOptionalDtype();
        if(elemType && elemType.isF64())
            elemType = Float32Type::get(context);
        
        diffusion_graph::ValueTensorType DGvTensor;
        DGvTensor = DGvTensor.get(context, 
                        torchvTensor.getOptionalSizes(), 
                        elemType, 
                        torchvTensor.getOptionalSparsity());
        opType = cast<Type>(DGvTensor);
        return opType;
    }

    else
        return type;
}

RankedTensorType mlir::diffusion_graph::convertTorchvTypeToTensorType(Type type){
    
    auto TorchTensor = cast<mlir::torch::Torch::ValueTensorType>(type);
    Type elemType = TorchTensor.getOptionalDtype();
    SmallVector<int64_t> shape = cast<SmallVector<int64_t>>(TorchTensor.getOptionalSizes());
    
    RankedTensorType tensor = RankedTensorType::get(shape, elemType);
    return tensor;
}

Type mlir::diffusion_graph::convertDGContainedType(Type type, 
                        MLIRContext *context){
    auto TorchList = cast<torch::Torch::ListType>(type);

    Type containedResultType;
    if(isa<torch::Torch::IntType>(TorchList.getContainedType()))
        containedResultType = IntegerType::get(context, 32);
    else if(isa<torch::Torch::FloatType>(TorchList.getContainedType()))
        containedResultType = Float32Type::get(context);
    else if(isa<torch::Torch::BoolType>(TorchList.getContainedType()))
        containedResultType = IntegerType::get(context, 1);
    else if(isa<torch::Torch::ValueTensorType>(TorchList.getContainedType())){
        containedResultType = diffusion_graph::convertTorchvTypeToDGvType(TorchList.getContainedType(), context);
    }
    else
        assert(false && "Type for the list not added");

    return containedResultType;
}

Type mlir::diffusion_graph::promoteDtype(Type a, Type b) {
  if (a == b) return a;

  if (isa<mlir::IntegerType>(a) && isa<mlir::IntegerType>(b)){
    auto int_a = cast<mlir::IntegerType>(a);
    auto int_b = cast<mlir::IntegerType>(b);
    if (int_a.getWidth() > int_b.getWidth()){
      return a;
    } else {
      return b;
    }
  }
  if (isa<mlir::Float32Type>(a) && isa<mlir::Float32Type>(b)){
    auto float_a = cast<mlir::Float32Type>(a);
    auto float_b = cast<mlir::Float32Type>(b);
    return a;
  }

  return Type();
}
