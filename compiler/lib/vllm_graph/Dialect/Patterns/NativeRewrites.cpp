#include "vllm_graph/Dialect/Patterns/NativeRewrites.hpp"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mlir::diffusion_graph;

bool mlir::diffusion_graph::createPoolingFunc(Value rootOpResult, PatternRewriter& rewriter){
    
    auto indexSelectOp = dyn_cast<diffusion_graph::IndexSelectOp>(rootOpResult.getDefiningOp());

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

        // Guard: if this result has no uses, treat it as terminal
        if (result.use_empty()) {
            newLastValue = clonedOp->getResult(0);
            oldLastValue = result;
            lastOp = nullptr;
            continue;
        }

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

    
    return true;

}

bool mlir::diffusion_graph::createCLIPPoolingFunc(Value rootOpResult, PatternRewriter& rewriter){
    
    auto indexSelectOp = dyn_cast<diffusion_graph::IndexSelectOp>(rootOpResult.getDefiningOp());

    if(!indexSelectOp){
        return false;
    }

    MLIRContext *context = indexSelectOp->getContext();
    func::FuncOp mainFunc = indexSelectOp->getParentOfType<func::FuncOp>();
    auto module = mainFunc->getParentOfType<ModuleOp>();


    Location loc = mainFunc.getLoc();

    Value arg = indexSelectOp.getSelf();
    Value indices = indexSelectOp.getIndices();

    auto maxDimOp = indices.getDefiningOp<diffusion_graph::MaxDimOp>();
    if(!maxDimOp){
        return false;
    }

    Value input = maxDimOp.getOperands()[0];

    auto castOp = input.getDefiningOp<diffusion_graph::DtypeCastOp>();
    if(!castOp){
        return false;
    }

    input = castOp.getOperands()[0];


    SmallVector<Type> inputTypes = {arg.getType(), input.getType()};

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

    Value input1 = indexSelectOp.getSelf();
    Value input2 = castOp.getOperands()[0];

    mapper.map(input1, entryBlock->getArgument(0));
    mapper.map(input2, entryBlock->getArgument(1));

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

        // Map ALL results of this op (handles multi-result ops like MaxDimOp)
        for (unsigned i = 0; i < currOp->getNumResults(); ++i)
            mapper.map(currOp->getResult(i), clonedOp->getResult(i));

        OpStack.pop_back();

        // Find the first result that actually has uses to drive traversal
        Value result;
        Value clonedResult;
        for (unsigned i = 0; i < currOp->getNumResults(); ++i) {
            if (!currOp->getResult(i).use_empty()) {
                result = currOp->getResult(i);
                clonedResult = clonedOp->getResult(i);
                break;
            }
        }

        // If no result has any uses, treat this op as terminal
        if (!result) {
            newLastValue = clonedOp->getResult(0);
            oldLastValue = currOp->getResult(0);
            lastOp = nullptr;
            continue;
        }

        Operation *nextOp = (*result.use_begin()).getOwner();
        if(!nextOp)
            continue;

        if(!isa<func::ReturnOp>(*nextOp) && OpStack.empty())
            OpStack.push_back(nextOp);

        else{
            newLastValue = clonedResult;
            oldLastValue = result;
            lastOp = nextOp;
        }
            
    }

    rewriter.create<func::ReturnOp>(loc, newLastValue);

    
    mlir::Block &oldEntryBlock = mainFunc.getBody().front();
    rewriter.setInsertionPointToEnd(&oldEntryBlock);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(lastOp, arg);

    return true;

}