#ifndef VLLM_GRAPH_DIALECT_PATTERNS_PATTERNINTERPRETER_H
#define VLLM_GRAPH_DIALECT_PATTERNS_PATTERNINTERPRETER_H

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

using namespace mlir;
using namespace mlir::pdl_interp;
using namespace mlir::pdl;

class PDLInterpMatcher {
private:
    std::unique_ptr<MLIRContext> context;
    OwningOpRef<ModuleOp> pdlModule;
    
    // Interpreter state for pattern matching
    struct InterpreterState {
        Operation* currentOp = nullptr;
        DenseMap<Value, Operation*> valueToOp;
        DenseMap<Value, Value> valueToValue;
        llvm::DenseMap<Value, SmallVector<Operation*, 4>> valueToOpRanges;
        llvm::DenseMap<Value, ValueRange> valueToValueRanges;
        DenseMap<Value, Attribute> valueToAttr;
        SmallVector<Operation*> matches;
        bool success = false;
    };
    
    // Execute a single PDL interpreter instruction
    bool executeInstruction(Operation* instrOp, InterpreterState& state);
    
    bool executeCheckOperationName(CheckOperationNameOp op, InterpreterState& state);
    
    bool executeCheckOperandCount(CheckOperandCountOp op, InterpreterState& state);

    bool executeCheckResultCount(CheckResultCountOp op, InterpreterState& state);
    
    bool executeGetOperand(GetOperandOp op, InterpreterState& state);

    bool executeGetOperands(GetOperandsOp op, InterpreterState& state);
    
    bool executeGetResult(GetResultOp op, InterpreterState& state);
    
    bool executeGetAttribute(GetAttributeOp op, InterpreterState& state);
    
    bool executeRecordMatch(RecordMatchOp op, InterpreterState& state);
    
    bool executeApplyRewrite(ApplyRewriteOp op, InterpreterState& state);

    bool executeIsNotNull(IsNotNullOp op, InterpreterState& state);

    bool executeAreEqual(AreEqualOp op, InterpreterState& state);

    bool executeGetUsers(GetUsersOp op, InterpreterState& state);

    bool executeForLoop(ForEachOp op, InterpreterState& state);
    
    bool executeBlock(Block& block, InterpreterState& state);
    
    bool executeRegion(Region& region, InterpreterState& state);
    
public:
    PDLInterpMatcher() {
        context = std::make_unique<MLIRContext>();
        context->loadDialect<pdl_interp::PDLInterpDialect>();
        context->loadDialect<pdl::PDLDialect>();
    }
    
    bool loadPDLInterpFile(const std::string& filename);
    
    bool matchPattern(Operation* rootOp);
    // Recursively match against IR tree
    bool matchInTree(Operation* op);
    
    void dumpPDLModule() {
        if (pdlModule) {
            pdlModule->dump();
        }
    }
};

#endif