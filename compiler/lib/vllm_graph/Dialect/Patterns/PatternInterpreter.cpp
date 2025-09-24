#include "vllm_graph/Dialect/Patterns/PatternInterpreter.hpp"

#include <memory>
#include <vector>

using namespace mlir::vllm_graph;

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
    if (auto getAttribute = dyn_cast<GetAttributeOp>(instrOp)) {
        return executeGetAttribute(getAttribute, state);
    }
    if (auto recordMatch = dyn_cast<RecordMatchOp>(instrOp)) {
        return executeRecordMatch(recordMatch, state);
    }
    if (auto applyRewrite = dyn_cast<ApplyRewriteOp>(instrOp)) {
        return executeApplyRewrite(applyRewrite, state);
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
    if (instrOp->hasTrait<OpTrait::IsTerminator>()) {
        return true; // Continue execution
    }
    
    llvm::errs() << instrOp->getName() << " is not implemented\n";

    return false; // Default: continue execution
}

bool PDLInterpMatcher::executeCheckOperationName(CheckOperationNameOp op, InterpreterState& state) const {
    if (!state.currentOp) return false;
    
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
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
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    
    if(inputOp->getNumOperands() == op.getCount())
        return executeBlock(*trueDest, state);
    else
        return executeBlock(*falseDest, state);

}

bool PDLInterpMatcher::executeGetDefiningOp(GetDefiningOpOp op, InterpreterState& state) const {
    
    Value inputValue = state.valueToValue.lookup(op.getInputOp());
    if (!inputValue) return false;

    Operation* def_op = inputValue.getDefiningOp();

    state.valueToOp[op.getValue()] = def_op;

    return true;
}

bool PDLInterpMatcher::executeCheckResultCount(CheckResultCountOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;

    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    if(inputOp->getNumResults() == op.getCount())
        return executeBlock(*trueDest, state);
    else
        return executeBlock(*falseDest, state);
}

bool PDLInterpMatcher::executeGetOperand(GetOperandOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    uint32_t index = op.getIndex();
    if (index >= inputOp->getNumOperands()) return false;
    
    Value operand = inputOp->getOperand(index);
    state.valueToValue[op.getValue()] = operand;
    
    return true;
}

bool PDLInterpMatcher::executeGetResult(GetResultOp op, InterpreterState& state) const {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    uint32_t index = op.getIndex();
    if (index >= inputOp->getNumResults()) return false;
    
    // Store the result value mapping
    Value result = inputOp->getResult(index);
    
    state.valueToValue[op.getValue()] = result;
    
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

    Block* SuccessorBlock = op.getSuccessor();
    BlockArgument arg = op.getLoopVariable();

    Region& bodyRegion = mlir::cast<pdl_interp::ForEachOp>(op).getRegion();
    Block* LoopBody = &bodyRegion.front();
    //TODO: Need to do it for Types and Attributes as well.
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
    Operation* inputOp = state.valueToOp.lookup(op.getOperand());
    SmallVector<Operation*, 4> OpRanges;
    for(Operation *user : inputOp->getUsers())
        OpRanges.push_back(user);

    state.valueToOpRanges[op.getValue()] = OpRanges;
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
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());

    state.valueToValueRanges[op.getValue()] = inputOp->getOperands();
    return true;
}

bool PDLInterpMatcher::executeRecordMatch(RecordMatchOp op, InterpreterState& state) const{
    // Record all matched operations
    for (Value input : op.getInputs()) {
        if (Operation* matchedOp = state.valueToOp.lookup(input)) {
            state.matches.push_back(matchedOp);
        }
    }
    state.success = true;

    return true;
}

bool PDLInterpMatcher::executeApplyRewrite(ApplyRewriteOp op, InterpreterState& state) const {
    // This would execute the actual rewrite
    // For now, just mark as successful match
    state.success = true;
    return true;
}

bool PDLInterpMatcher::executeBlock(Block& block, InterpreterState& state) const {
    for (Operation& op : block) {
        llvm::outs() << op << "\n";
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
                llvm::outs() << nestedOp << "\n";
                if (matchInTree(&nestedOp)) {
                    return true;
                }
            }
        }
    }

    return false;
    
}

