#include "vllm_graph/Dialect/Patterns/NativeRewrites.hpp"

using namespace mlir;
using namespace mlir::vllm_graph;

bool mlir::vllm_graph::createPoolingFunc(Value rootOpResult, PatternRewriter& rewriter){
    
    auto indexSelectOp = dyn_cast<vllm_graph::IndexSelectOp>(rootOpResult.getDefiningOp());

    if(!indexSelectOp){
        llvm::outs() << __LINE__ << " " << __FILE__ << "\n";
        return false;
    }
    MLIRContext *context = indexSelectOp->getContext();
    func::FuncOp mainFunc = indexSelectOp->getParentOfType<func::FuncOp>();
    auto module = mainFunc->getParentOfType<ModuleOp>();

    Location loc = mainFunc.getLoc();

    Value arg = indexSelectOp.getSelf();
    SmallVector<Type> inputTypes = {arg.getType()};

    FunctionType funcType = mainFunc.getFunctionType();
    SmallVector<Type> resultTypes = {}; //funcType.getResults()[1] 

    rewriter.setInsertionPoint(module.getBody(), module.getBody()->end());

    auto newFunc = rewriter.create<func::FuncOp>(
        loc, 
        "compute_pooling_layer",  // Function name
        funcType
    );

    Block *entryBlock = rewriter.createBlock(&newFunc.getBody());
    for (Type argType : inputTypes) {
      entryBlock->addArgument(argType, loc);
    }

    // Set insertion point to the entry block to add operations
    rewriter.setInsertionPointToStart(entryBlock);

    llvm::outs() << module << "\n";

    // OpBuilder builder(context);

    // mlir::FunctionType poolingFuncType = builder.getFunctionType(inputType, resultType);

    // auto PoolingFuncOp = builder.create<func::FuncOp>(
    //   loc, "compute_pooling_layer", poolingFuncType);

    // Block *entryBlock = PoolingFuncOp.addEntryBlock();
  
    // // Set insertion point to the new function's entry block
    // OpBuilder::InsertionGuard guard(builder);
    // builder.setInsertionPointToStart(entryBlock);

    // llvm::outs() << PoolingFuncOp << "\n";
    return true;
    // IRMapper mapping;
    // 


    // 
    // Value value = rootOpResult;

    // while(!value){
    //     OpOperand &use : value.getUses()[0];
    //     Operation *userOp = use.getOwner();
        

    //     value = userOp->getResult();
    // }

}