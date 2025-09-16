#include "vllm_graph/Dialect/Patterns/Patterns/PatternInterpreter.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <vector>


bool PDLInterpMatcher::executeInstruction(Operation* instrOp, InterpreterState& state) {
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

    if (auto finalize = dyn_cast<FinalizeOp>(instrOp)) {
        state.success = true;
        return true;
    }
    
    // Handle control flow operations
    if (instrOp->hasTrait<OpTrait::IsTerminator>()) {
        return true; // Continue execution
    }
    
    return true; // Default: continue execution
}

bool PDLInterpMatcher::executeCheckOperationName(CheckOperationNameOp op, InterpreterState& state) {
    if (!state.currentOp) return false;
    
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    StringRef expectedName = op.getName();
    return inputOp->getName().getStringRef() == expectedName;
}

bool PDLInterpMatcher::executeCheckOperandCount(CheckOperandCountOp op, InterpreterState& state) {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    return inputOp->getNumOperands() == op.getCount();
}

bool PDLInterpMatcher::executeCheckResultCount(CheckResultCountOp op, InterpreterState& state) {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    return inputOp->getNumResults() == op.getCount();
}

bool PDLInterpMatcher::executeGetOperand(GetOperandOp op, InterpreterState& state) {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    uint32_t index = op.getIndex();
    if (index >= inputOp->getNumOperands()) return false;
    
    Value operand = inputOp->getOperand(index);
    if (Operation* definingOp = operand.getDefiningOp()) {
        state.valueToOp[op.getValue()] = definingOp;
    }
    
    return true;
}

bool PDLInterpMatcher::executeGetResult(GetResultOp op, InterpreterState& state) {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    uint32_t index = op.getIndex();
    if (index >= inputOp->getNumResults()) return false;
    
    // Store the result value mapping
    Value result = inputOp->getResult(index);
    // For this context, we can map the result back to its defining operation
    state.valueToOp[op.getValue()] = inputOp;
    
    return true;
}

bool PDLInterpMatcher::executeGetAttribute(GetAttributeOp op, InterpreterState& state) {
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());
    if (!inputOp) return false;
    
    StringRef attrName = op.getName();
    if (Attribute attr = inputOp->getAttr(attrName)) {
        state.valueToAttr[op.getAttribute()] = attr;
        return true;
    }
    
    return false;
}

bool PDLInterpMatcher::executeIsNotNull(IsNotNullOp op, InterpreterState& state){
    Operation* inputOp = state.valueToOp.lookup(op.getInputOp());

    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    if(inputOp)
        return executeBlock(*trueDest, state);
    
    else
        return executeBlock(*falseDest, state);
}

bool PDLInterpMatcher::executeRecordMatch(RecordMatchOp op, InterpreterState& state) {
    // Record all matched operations
    for (Value input : op.getInputs()) {
        if (Operation* matchedOp = state.valueToOp.lookup(input)) {
            state.matches.push_back(matchedOp);
        }
    }
    return true;
}

bool PDLInterpMatcher::executeApplyRewrite(ApplyRewriteOp op, InterpreterState& state) {
    // This would execute the actual rewrite
    // For now, just mark as successful match
    state.success = true;
    return true;
}

bool PDLInterpMatcher::executeBlock(Block& block, InterpreterState& state){
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

bool PDLInterpMatcher::executeRegion(Region& region, InterpreterState& state) {
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
    
    pdlModule = mlir::parseSourceFile<ModuleOp>(sourceMgr, context.get());
    if (!pdlModule) {
        llvm::errs() << "Failed to parse PDL-interp file\n";
        return false;
    }
    
    return true;
}

bool PDLInterpMatcher::matchPattern(Operation* rootOp) {
    if (!pdlModule) {
        llvm::errs() << "No PDL-interp module loaded\n";
        return false;
    }
    
    // Initialize interpreter state
    InterpreterState state;
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
    return executeRegion(mainFunc.getBody(), state);
}

// Recursively match against IR tree
bool PDLInterpMatcher::matchInTree(Operation* op) {
    // Try matching at current operation
    if (matchPattern(op)) {
        return true;
    }
    
    // Recursively check nested operations
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
