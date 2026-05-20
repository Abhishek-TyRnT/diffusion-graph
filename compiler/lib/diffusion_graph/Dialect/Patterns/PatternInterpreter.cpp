#include "diffusion_graph/Dialect/Patterns/PatternInterpreter.hpp"
#include "llvm/Support/Compiler.h"

#include <memory>
#include <vector>

using namespace mlir::diffusion_graph;

bool PDLInterpMatcher::executeInstruction(Operation* instrOp, InterpreterState& state) const {
    if (auto checkOpName = dyn_cast<CheckOperationNameOp>(instrOp)) {
        return executeCheckOperationName(checkOpName, state);
    }
    if (auto checkOperandCount = dyn_cast<CheckOperandCountOp>(instrOp)) {
        return executeCheckOperandCount(checkOperandCount, state);
    }
    if (auto checkResultCount = dyn_cast<CheckResultCountOp>(instrOp)) {
        return executeCheckResultCount(checkResultCount, state);
    }
    if (auto getOperand = dyn_cast<GetOperandOp>(instrOp)) {
        return executeGetOperand(getOperand, state);
    }
    if (auto getResult = dyn_cast<GetResultOp>(instrOp)) {
        return executeGetResult(getResult, state);
    }
    if (auto getResults = dyn_cast<GetResultsOp>(instrOp)) {
        return executeGetResults(getResults, state);
    }
    if (auto extractOp = dyn_cast<ExtractOp>(instrOp)) {
        return executeExtractOp(extractOp, state);
    }
    if (auto getAttribute = dyn_cast<GetAttributeOp>(instrOp)) {
        return executeGetAttribute(getAttribute, state);
    }
    if (auto recordMatch = dyn_cast<RecordMatchOp>(instrOp)) {
        return executeRecordMatch(recordMatch, state);
    }

    if (auto isNotNull = dyn_cast<IsNotNullOp>(instrOp)) {
        return executeIsNotNull(isNotNull, state);
    }

    if (auto forEach = dyn_cast<ForEachOp>(instrOp)) {
        return executeForLoop(forEach, state); 
    }

    if (auto areEqual = dyn_cast<AreEqualOp>(instrOp)) {
        return executeAreEqual(areEqual, state);
    }

    if (auto getOperands = dyn_cast<GetOperandsOp>(instrOp)) {
        return executeGetOperands(getOperands, state);
    }

    if ( auto getUsers = dyn_cast<GetUsersOp>(instrOp)) {
        return executeGetUsers(getUsers, state);
    }

    if ( auto getDefOp = dyn_cast<GetDefiningOpOp>(instrOp)) {
        return executeGetDefiningOp(getDefOp, state);
    }

    if (auto finalize = dyn_cast<FinalizeOp>(instrOp)) {
        return true;
    }

    if (auto finalize = dyn_cast<ContinueOp>(instrOp)) {
        return true;
    }
    
    // Handle control flow operations
    if (instrOp->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return true; // Continue execution
    }
    
    llvm::errs() << instrOp->getName() << " is not implemented\n";
    assert(false && "Fatal Interpretor error");

    return false;
}

bool PDLInterpMatcher::executeCheckOperationName(CheckOperationNameOp op, InterpreterState& state) const {
    if (!state.currentOp) return false;
    
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());
    if (!inputOp) return false;
    
    StringRef expectedName = op.getName();

    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    if(inputOp->getName().getStringRef() == expectedName)
        return executeBlock(*trueDest, state);
    else
        return executeBlock(*falseDest, state);
}

bool PDLInterpMatcher::executeCheckOperandCount(CheckOperandCountOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());
    if (!inputOp) return false;
    
    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    
    if(inputOp->getNumOperands() == op.getCount())
        return executeBlock(*trueDest, state);
    else
        return executeBlock(*falseDest, state);

}

bool PDLInterpMatcher::executeGetDefiningOp(GetDefiningOpOp op, InterpreterState& state) const {
    
    Value inputValue = state.valueToValue.lookup(op.getOperand());
    if (!inputValue) return false;

    Operation* def_op = inputValue.getDefiningOp();

    state.valueToOp[op.getInputOp()] = def_op;

    return true;
}

bool PDLInterpMatcher::executeCheckResultCount(CheckResultCountOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());
    if (!inputOp) return false;

    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    if(inputOp->getNumResults() == op.getCount())
        return executeBlock(*trueDest, state);
    else
        return executeBlock(*falseDest, state);
}

bool PDLInterpMatcher::executeGetOperand(GetOperandOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());
    if (!inputOp) return false;
    
    uint32_t index = op.getIndex();
    if (index >= inputOp->getNumOperands()) return false;
    
    Value operand = inputOp->getOperand(index);

    state.valueToValue[op.getResult()] = operand;
    
    return true;
}

bool PDLInterpMatcher::executeGetResult(GetResultOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());
    if (!inputOp) return false;
    
    uint32_t index = op.getIndex();
    if (index >= inputOp->getNumResults()) return false;
    
    // Store the result value mapping
    Value result = inputOp->getResult(index);
    state.valueToValue[op.getResult()] = result;
    
    return true;
}

bool PDLInterpMatcher::executeGetResults(GetResultsOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());

    if(!inputOp)
        return false;

    Type type = op.getValue().getType();
    ValueRange results = inputOp->getResults();
    
    if(isa<RangeType>(type)){
        state.valueToValueRanges[op.getResult()] = results;
    } else {
        uint32_t index = op.getIndex().value();
        if (index >= results.size()) return false;
        Value result = results[index];
        state.valueToValue[op.getResult()] = result;
    }
    return true;
}

bool PDLInterpMatcher::executeExtractOp(ExtractOp op, InterpreterState& state) const {
    ValueRange valueRange = state.valueToValueRanges.lookup(op.getOperand());

    int32_t index = op.getIndex();
    if(index >= valueRange.size())
        return false;

    Value value = valueRange[index];
    state.valueToValue[op.getResult()] = value;

    return true;
}

bool PDLInterpMatcher::executeGetAttribute(GetAttributeOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    StringRef attrName = op.getName();
    if (Attribute attr = inputOp->getAttr(attrName)) {
        state.valueToAttr[op.getAttribute()] = attr;
        return true;
    }
    
    return false;
}

bool PDLInterpMatcher::executeIsNotNull(IsNotNullOp op, InterpreterState& state) const {

    Type type = op.getValue().getType();
    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();
    if(isa<OperationType>(type)){
        Operation* inputOp = state.valueToOp.lookup(op.getOperand());

        if(inputOp)
            return executeBlock(*trueDest, state);
        else
            return executeBlock(*falseDest, state);
    } else {
        Value input = state.valueToValue.lookup(op.getOperand());
        if(input)
            return executeBlock(*trueDest, state);
        else
            return executeBlock(*falseDest, state);
    }
}

bool PDLInterpMatcher::executeForLoop(ForEachOp op, InterpreterState& state) const {
    
    SmallVector<Operation*, 4> OpRanges = state.valueToOpRanges.lookup(op.getOperand());

    if(OpRanges.size() == 0)
        return false;

    Block* SuccessorBlock = op.getSuccessor();
    BlockArgument arg = op.getLoopVariable();

    Region& bodyRegion = mlir::cast<pdl_interp::ForEachOp>(op).getRegion();
    Block* LoopBody = &bodyRegion.front();
    
    for(Operation* op : OpRanges){
        state.valueToOp[arg] = op;
        bool passed = executeBlock(*LoopBody, state);
        if(state.success)
            break;

        if(!passed)
            return false;
    }

    return executeBlock(*SuccessorBlock, state);
}

bool PDLInterpMatcher::executeGetUsers(GetUsersOp op, InterpreterState& state) const {

    Type type = op.getValue().getType();
    SmallVector<Operation*, 4> OpRanges;
    if(isa<OperationType>(type)){
        Operation* inputOp = state.valueToOp.lookup(op.getOperand());
        if(!inputOp)
            return false;
        for(Operation *user : inputOp->getUsers())
            OpRanges.push_back(user);
    } else {
        Value input = state.valueToValue.lookup(op.getOperand());
        if(!input)
            return false;

        for(Operation *user : input.getUsers())
            OpRanges.push_back(user);
    }

    state.valueToOpRanges[op.getResult()] = OpRanges;
    return true;
    
}

bool PDLInterpMatcher::executeAreEqual(AreEqualOp op, InterpreterState& state) const{
    Type type = op.getOperands()[0].getType();
    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();
    if(isa<OperationType>(type)){
        if(state.valueToOp.lookup(op.getLhs()) == state.valueToOp.lookup(op.getRhs()))
            return executeBlock(*trueDest, state);
        else
            return executeBlock(*falseDest, state);
    } else if(isa<RangeType>(type)) {
        //TODO: Implement OperationRange and AttributeRange
        RangeType rangeType = cast<RangeType>(type);
        if(isa<ValueType>(rangeType.getElementType())){
            Value lhskey = op.getLhs();
            Value rhskey = op.getRhs();
            ValueRange lhs(state.valueToValue.lookup(lhskey));
            ValueRange rhs(state.valueToValue.lookup(rhskey));

            if (lhs.size() != rhs.size())
                return executeBlock(*falseDest, state);
            for (auto [l, r] : llvm::zip(lhs, rhs)) {
                if (l != r) 
                    return executeBlock(*falseDest, state);
            }
            return executeBlock(*trueDest, state);
        } else 
            return false;

    } else {
        
        if(state.valueToValue.lookup(op.getLhs()) == state.valueToValue.lookup(op.getRhs()))
            return executeBlock(*trueDest, state);
        else
            return executeBlock(*falseDest, state);
    }
    
}

bool PDLInterpMatcher::executeGetOperands(GetOperandsOp op, InterpreterState& state) const{
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());

    state.valueToValueRanges[op.getResult()] = inputOp->getOperands();
    return true;
}

bool PDLInterpMatcher::executeRecordMatch(RecordMatchOp op, InterpreterState& state) const{
    // Record all matched operations
    for (Value input : op.getInputs()) {
        if (Operation* matchedOp = state.valueToOp.lookup(input)) {
            rewriteState->matches.push_back(matchedOp);
        }
    }

    rewriteState->rewriterSymbol = op.getRewriter();
    state.success = true;

    return true;
}

bool PDLInterpMatcher::executeApplyRewrite(ApplyRewriteOp op, PatternRewriter& rewriter, InterpreterState& state) const {
    // TODO: make it type agnostic.
    std::string nativeRewriteName = op.getName().str();
    Value input = state.valueToValue.lookup(op.getOperand(0));
    if(holds_alternative<function<bool(Value, PatternRewriter&)>>(functionMap.at(nativeRewriteName))){
        auto func = get<function<bool(Value, PatternRewriter&)>>(functionMap.at(nativeRewriteName));
        return func(input, rewriter);
    }
    
    return false;
}

bool PDLInterpMatcher::executeBlock(Block& block, InterpreterState& state) const {
    for (Operation& op : block) {

        if (!executeInstruction(&op, state)) {
            return false;
        }
        
        // If we hit a successful match, we can stop
        if (state.success) {
            return true;
        }
    }

    return true;
}

bool PDLInterpMatcher::executeEraseOp(EraseOp op, PatternRewriter& rewriter, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());
    if(!inputOp)
        return false;

    Value input = inputOp->getOperand(0);
    Value result = inputOp->getResult(1);

    if(!inputOp->use_empty())
        return false;

    rewriter.eraseOp(inputOp);
    
    return true;
}

bool PDLInterpMatcher::executeRegion(Region& region, InterpreterState& state) const{
    if (region.empty()) return true;
    
    Block& block = region.front();
    
    return executeBlock(block, state);
}

bool PDLInterpMatcher::loadPDLInterpFile(const std::string& filename) {
        // Read the file
    std::string errorMessage;
    auto file = mlir::openInputFile(filename, &errorMessage);
    if (!file) {
        llvm::errs() << "Error opening file: " << errorMessage << "\n";
        return false;
    }
    
    // Parse the PDL-interp module
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    
    pdlModule = mlir::parseSourceFile<ModuleOp>(sourceMgr, context);
    if (!pdlModule) {
        llvm::errs() << "Failed to parse PDL-interp file\n";
        return false;
    }
    
    return true;
}

bool PDLInterpMatcher::matchPattern(Operation* rootOp) const {

    InterpreterState state;

    if (!pdlModule) {
        llvm::errs() << "No PDL-interp module loaded\n";
        return false;
    }
    
    // Initialize interpreter state
    
    state.currentOp = rootOp;
    // Find the main matching function (typically the first function)
    auto funcs = pdlModule->getOps<pdl_interp::FuncOp>();
    if (funcs.empty()) {
        llvm::errs() << "No PDL-interp functions found\n";
        return false;
    }
    
    pdl_interp::FuncOp mainFunc = *funcs.begin();
    
    // Set up initial bindings - map root operation
    Block& entryBlock = mainFunc.getBody().front();
    if (!entryBlock.getArguments().empty()) {
        Value rootArg = entryBlock.getArgument(0);
        state.valueToOp[rootArg] = rootOp;
    }    
    // Execute the function
    if(!executeRegion(mainFunc.getBody(), state))
        return false;

    return state.success;
}

// Recursively match against IR tree
bool PDLInterpMatcher::matchInTree(Operation* op) const {
    // Try matching at current operation
    if(matchPattern(op))
        return true;
    
    // Recursively check nested operations
    //TODO: Add this feature later on
    for (Region& region : op->getRegions()) {
        for (Block& block : region) {
            for (Operation& nestedOp : block) {

                if (matchInTree(&nestedOp)) {
                    return true;
                }
            }
        }
    }

    return false;
    
}

bool PDLInterpMatcher::rewriteOrExecuteOps(Operation* op, PatternRewriter& rewriter, InterpreterState& state) const {

    if(auto applyRewriteOp = dyn_cast<ApplyRewriteOp>(op)){
        return executeApplyRewrite(applyRewriteOp, rewriter, state);
    } else if(auto eraseOp = dyn_cast<EraseOp>(op)){
        return executeEraseOp(eraseOp, rewriter, state);
    } else {
        return executeInstruction(op, state);
    }
}

LogicalResult PDLInterpMatcher::rewriteModule(PatternRewriter& rewriter) const {

    InterpreterState state;

    if (!pdlModule) {
        llvm::errs() << "No PDL-interp module loaded\n";
        return failure();
    }
    
    // Initialize interpreter state
    
    // state.currentOp = rootOp;
    
    // Find the main matching function (typically the first function)
    SymbolRefAttr sym = rewriteState->rewriterSymbol;
    Operation* symbolOp = SymbolTable::lookupNearestSymbolFrom(*pdlModule, sym);

    if (!symbolOp) {
        // Symbol not found
        llvm::errs() << "Symbol not found: " << sym;
        return failure();
    }

    auto rewriterModule = mlir::dyn_cast<pdl_interp::FuncOp>(symbolOp);

    if(!rewriterModule){
        llvm::errs() << "Rewriter Module not found\n";
        return failure();
    }
    
    // // Set up initial bindings - map root operation
    Block& entryBlock = rewriterModule.getBody().front();
    int i = 0;
    for (BlockArgument arg: entryBlock.getArguments()) {
        state.valueToOp[arg] = rewriteState->matches[i++];
    }    
    // Execute the function
    for(Operation& rewriteOp : entryBlock){
        bool passed = rewriteOrExecuteOps(&rewriteOp, rewriter, state);
        
        if(!passed)
            return failure();
    }

    return success();
}

void PDLInterpMatcher::registerNativeRewrites() {
    //Register more native rewrites as and when they are required
    functionMap["CreatePoolingFunc"] = std::function<bool(Value, PatternRewriter&)>(createPoolingFunc);
    functionMap["CreateCLIPPoolingFunc"] = std::function<bool(Value, PatternRewriter&)>(createCLIPPoolingFunc);
}

