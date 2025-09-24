#ifndef VLLM_GRAPH_DIALECT_PATTERNS_PATTERNINTERPRETER_H
#define VLLM_GRAPH_DIALECT_PATTERNS_PATTERNINTERPRETER_H

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pdl_interp;
using namespace mlir::pdl;

namespace mlir {

namespace vllm_graph {
class PDLInterpMatcher {
private:
    MLIRContext *context;
    mutable OwningOpRef<ModuleOp> pdlModule;
    
    // Interpreter state for pattern matching
    struct InterpreterState {
        Operation* currentOp = nullptr;
        DenseMap<Value, Operation*> valueToOp;
        DenseMap<Value, Value> valueToValue;
        llvm::DenseMap<Value, SmallVector<Operation*, 4>> valueToOpRanges;
        llvm::DenseMap<Value, ValueRange> valueToValueRanges;
        DenseMap<Value, Attribute> valueToAttr;
        SmallVector<Operation*> matches;
        SymbolRefAttr rewriteCode;
        bool success = false;
    };
    
    // Execute a single PDL interpreter instruction
    bool executeInstruction(Operation* instrOp, InterpreterState& state) const;
    
    bool executeCheckOperationName(CheckOperationNameOp op, InterpreterState& state) const;
    
    bool executeCheckOperandCount(CheckOperandCountOp op, InterpreterState& state) const;

    bool executeCheckResultCount(CheckResultCountOp op, InterpreterState& state) const;
    
    bool executeGetOperand(GetOperandOp op, InterpreterState& state) const;

    bool executeGetOperands(GetOperandsOp op, InterpreterState& state) const;
    
    bool executeGetResult(GetResultOp op, InterpreterState& state) const ;
    
    bool executeGetAttribute(GetAttributeOp op, InterpreterState& state) const;
    
    bool executeRecordMatch(RecordMatchOp op, InterpreterState& state) const;
    
    bool executeApplyRewrite(ApplyRewriteOp op, InterpreterState& state) const;

    bool executeIsNotNull(IsNotNullOp op, InterpreterState& state) const;

    bool executeAreEqual(AreEqualOp op, InterpreterState& state) const;

    bool executeGetUsers(GetUsersOp op, InterpreterState& state) const;

    bool executeGetDefiningOp(GetDefiningOpOp op, InterpreterState& state) const;

    bool executeForLoop(ForEachOp op, InterpreterState& state) const;
    
    bool executeBlock(Block& block, InterpreterState& state) const;
    
    bool executeRegion(Region& region, InterpreterState& state) const;
    
    bool loadPDLInterpFile(const std::string& filename);
public:
    PDLInterpMatcher(MLIRContext *context, const std::string& pdlFile) {
        this->context = context;
        loadPDLInterpFile(pdlFile);
            
    }
    
    bool matchPattern(Operation* rootOp) const;
    // Recursively match against IR tree
    bool matchInTree(Operation* op) const;
    
    void dumpPDLModule() {
        if (pdlModule) {
            pdlModule->dump();
        }
    }
};

} // mlir
} // vllm_graph
#endif