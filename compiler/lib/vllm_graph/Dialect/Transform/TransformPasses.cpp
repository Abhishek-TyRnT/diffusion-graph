
#include "PassDetail.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "vllm_graph/Dialect/IR/DiffusionGraphTypes.hpp"
#include "vllm_graph/Dialect/IR/DiffusionGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/DiffusionGraphOps.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "vllm_graph/Utils/Utils.hpp"

#include <vector>

using namespace mlir;
using namespace mlir::diffusion_graph;

//TODO: Unify type conversion across all passes
class vLLMGraphConversion : public TypeConverter {
private:

public:
    vLLMGraphConversion(MLIRContext *context) {
        // Add conversions for primitive types
        
        addConversion([](Type type) -> std::optional<Type> {
            // Pass-through unchanged types
            return type;
        });

        // Convert f32 to f64
        addConversion([this, context](torch::Torch::ValueTensorType type) -> std::optional<Type> {
            return convertTorchvTypeToDGvType(type, context);      
        });

        addConversion([context](torch::Torch::FloatType type) -> std::optional<Type> {
            return Float32Type::get(context);
        });

        addConversion([context](torch::Torch::IntType type) -> std::optional<Type> {
            return IntegerType::get(context, 32);
        });

        addConversion([context](torch::Torch::StringType type) -> std::optional<Type> {
            return diffusion_graph::StringType::get(context);
        });

        addConversion([context](torch::Torch::BoolType type) -> std::optional<Type> {
            return IntegerType::get(context, 1);
        });

        addConversion([context](torch::Torch::ListType type) -> std::optional<Type> {
            auto containedType = convertDGContainedType(type, context);
            return diffusion_graph::ListType::get(context, containedType);
        });

        addConversion([context](torch::Torch::NoneType type) -> std::optional<Type> {
            return diffusion_graph::NoneType::get(context);
        });

    }

};

void replaceFuncDtypes(func::FuncOp &op, vLLMGraphConversion &typeConverter)
{
    /*
    The function replaces the torch.dtypes vllm_graph dtypes. It also adds a
    temporary cast Op, so as to not disturb Ops that have been transformed.

    NOTE: Currently only tested for single return types.
    
    */

    //Need to remove args which do not contribute to the
    //graph 
    mlir::Block &entryBlock = op.getBody().front();
    std::vector<uint32_t> removedArgIndices;

    for(unsigned i = 0; i < op.getNumArguments(); ++i){
        mlir::Value arg = op.getArgument(i);
        if(arg.use_empty()){
            arg.replaceAllUsesWith(nullptr);
            entryBlock.eraseArgument(i);
            removedArgIndices.push_back(i);
        }
    }
    mlir::FunctionType oldFuncType = op.getFunctionType();
    MLIRContext *context = op.getContext();
    mlir::OpBuilder builder(context);

    // Get current argument and result types
    llvm::ArrayRef<mlir::Type> oldArgTypes = oldFuncType.getInputs();
    llvm::ArrayRef<mlir::Type> oldResultTypes = oldFuncType.getResults();

    
    llvm::SmallVector<mlir::Type, 4> newArgTypes;


    //Making a list of dtype conversions from old torch dtypes to new vllm_graph_dtypes.
    uint32_t arg_index = 0;
    for (mlir::Type argType : oldArgTypes) {
        if(std::find(removedArgIndices.begin(), removedArgIndices.end(), arg_index) != removedArgIndices.end()){
            arg_index++;            
            continue;
        }
        auto newargType = typeConverter.convertType(argType);

        if(newargType)
            newArgTypes.push_back(newargType);
        else
            newArgTypes.push_back(argType);
        arg_index++;
    }

    llvm::SmallVector<mlir::Type, 4> newResultTypes;
    for (mlir::Type argType : oldResultTypes) {
        auto newargType = typeConverter.convertType(argType);
        if(newargType)
            newResultTypes.push_back(newargType);
        else
            newResultTypes.push_back(argType);  
    }

    // Function Type with new dtypes
    mlir::FunctionType newFuncType = builder.getFunctionType(newArgTypes, newResultTypes);

    op.setType(newFuncType);
    // mlir::Block &entryBlock = op.getBody().front();
    builder.setInsertionPointToStart(&entryBlock);

    /*Inserting an CastOp so as to not disturb subsequent Ops. 
    Note that in subsequent passes the cast Ops will be eliminated
    */
    //Adding offset so as to ignore types that have been removed.
    int offset = 0;
    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
        mlir::Value arg = op.getArgument(i);
        mlir::Value::user_range opList = arg.getUsers();
        mlir::Location loc = mlir::UnknownLoc::get(context);
        if(std::find(removedArgIndices.begin(), removedArgIndices.end(), i) != removedArgIndices.end())
            offset += 1;
        auto castOp = builder.create<diffusion_graph::CastOp>(loc, oldArgTypes[i + offset], arg);
        mlir::Operation *genCastOp = castOp.getOperation();
        mlir::Value castOpResult = castOp.getResult();
        arg.replaceAllUsesExcept(castOpResult, genCastOp);
        arg.setType(newArgTypes[i]);
    }


    //Same logic for return types.
    
    for(mlir::Operation &currOp : entryBlock)
    {
        if(mlir::isa<func::ReturnOp>(currOp))
        {
            auto returnOp = mlir::cast<func::ReturnOp>(currOp);
            mlir::Operation *returnOperation = returnOp.getOperation();
            for(int i =0 ; i < returnOperation->getNumOperands(); i++)
            {
                mlir::Value returnValue = returnOperation->getOperand(i);
                builder.setInsertionPoint(returnOp);
                mlir::Location returnloc = returnOp.getLoc();
                auto returnCastOp = builder.create<diffusion_graph::CastOp>(returnloc, newResultTypes[i], returnValue);
                returnOperation->setOperand(i, returnCastOp);
            }
            
        }
    }
}
LogicalResult convertFuncOp(ModuleOp &moduleOp){
    vLLMGraphConversion typeConverter(moduleOp.getContext());
    for (auto &op : moduleOp.getBody()->getOperations()) {
        if(auto funcOp = dyn_cast<func::FuncOp>(op))
        {
            replaceFuncDtypes(funcOp, typeConverter);
            
            return mlir::success();
        }
    }

    return mlir::failure();
}



namespace {
class ConvertGlobalTorchGraph : public ConvertGlobalFunctionPassBase<ConvertGlobalTorchGraph> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<diffusion_graph::DiffusionGraphIRDialect>();
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

std::unique_ptr<OperationPass<ModuleOp>> mlir::diffusion_graph::createConvertGlobalFunctionPass(){
    return std::make_unique<ConvertGlobalTorchGraph>();
}



