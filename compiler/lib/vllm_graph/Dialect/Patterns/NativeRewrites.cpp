#include "vllm_graph/Dialect/Patterns/NativeRewrites.hpp"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mlir::vllm_graph;

bool mlir::vllm_graph::createPoolingFunc(Value rootOpResult, PatternRewriter& rewriter){
    
    auto indexSelectOp = dyn_cast<vllm_graph::IndexSelectOp>(rootOpResult.getDefiningOp());

    if(!indexSelectOp){
        return false;
    }

    MLIRContext *context = indexSelectOp->getContext();
    func::FuncOp mainFunc = indexSelectOp->getParentOfType<func::FuncOp>();
    auto module = mainFunc->getParentOfType<ModuleOp>();

    Location loc = mainFunc.getLoc();

    Value arg = indexSelectOp.getSelf();
    SmallVector<Type> inputTypes = {arg.getType()};

    FunctionType funcType = mainFunc.getFunctionType();
    SmallVector<Type> newResultTypes = {funcType.getResults()[1] }; 
    SmallVector<Type> oldResultTypes = {funcType.getResults()[0] };

    FunctionType PoolingFuncType = rewriter.getFunctionType(inputTypes, newResultTypes);
    FunctionType updatedFuncTionType = rewriter.getFunctionType(funcType.getInputs(), oldResultTypes);

    rewriter.setInsertionPoint(module.getBody(), module.getBody()->end());

    auto newFunc = rewriter.create<func::FuncOp>(
        loc, 
        "compute_pooling_layer",  // Function name
        PoolingFuncType
    );

    mainFunc.setType(updatedFuncTionType);

    Block *entryBlock = rewriter.createBlock(&newFunc.getBody());
    for (Type argType : inputTypes) {
      entryBlock->addArgument(argType, loc);
    }

    // Set insertion point to the entry block to add operations
    rewriter.setInsertionPointToStart(entryBlock);

    IRMapping mapper;

    Value input = indexSelectOp.getSelf();

    mapper.map(input, entryBlock->getArgument(0));

    //Clone the Operations
    Value newLastValue, oldLastValue;
    Operation *lastOp; 
    SmallVector<Operation*> OpStack = {indexSelectOp};
    while(!OpStack.empty()){
        Operation* currOp = OpStack.back();
        bool continueFlag = false;
        for(Value operand : currOp->getOperands()){
            if(!mapper.contains(operand)){
                Operation* prevOp = operand.getDefiningOp();
                OpStack.push_back(prevOp);
                continueFlag = true;
                break;
            }
        }
        if(continueFlag)
            continue;
        
        Operation *clonedOp = rewriter.clone(*currOp, mapper);
        Value result = currOp->getResult(0);
        mapper.map(result, clonedOp->getResult(0));
        OpStack.pop_back();
        Operation *nextOp = (*result.use_begin()).getOwner();
        if(!nextOp)
            continue;

        if(!isa<func::ReturnOp>(*nextOp) && OpStack.empty())
            OpStack.push_back(nextOp);

        else{
            newLastValue = clonedOp->getResult(0);
            oldLastValue = result;
            lastOp = nextOp;
        }
            
    }

    rewriter.create<func::ReturnOp>(loc, newLastValue);

    
    mlir::Block &oldEntryBlock = mainFunc.getBody().front();
    rewriter.setInsertionPointToEnd(&oldEntryBlock);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(lastOp, arg);

    // for(Operation& op : oldEntryBlock){
    //     if(isa<func::ReturnOp>(op))
            
    // }
    
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