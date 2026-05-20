#ifndef DIFFUSION_GRAPH_DIALECT_PATTERNS_PATTERNINTERPRETER_H
#define DIFFUSION_GRAPH_DIALECT_PATTERNS_PATTERNINTERPRETER_H

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
#include "llvm/ADT/DenseMap.h"

#include "diffusion_graph/Dialect/Patterns/NativeRewrites.hpp"

using namespace mlir;
using namespace mlir::pdl_interp;
using namespace mlir::pdl;
using namespace std;

namespace mlir {

namespace diffusion_graph {
class PDLInterpMatcher {

using FuncVariant = std::variant<
    std::function<bool(PatternRewriter&)>,
    std::function<bool(Value, PatternRewriter&)>
>;

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
        bool success = false;
    };

    struct RewriteState {
        SmallVector<Operation*> matches;
        SymbolRefAttr rewriterSymbol;
        LogicalResult passed = failure();
    };

    std::unique_ptr<RewriteState> rewriteState;
    std::unordered_map<std::string, FuncVariant> functionMap;
    
    // Execute a single PDL interpreter instruction
    bool executeInstruction(Operation* instrOp, InterpreterState& state) const;
    
    bool executeCheckOperationName(CheckOperationNameOp op, InterpreterState& state) const;
    
    bool executeCheckOperandCount(CheckOperandCountOp op, InterpreterState& state) const;

    bool executeCheckResultCount(CheckResultCountOp op, InterpreterState& state) const;
    
    bool executeGetOperand(GetOperandOp op, InterpreterState& state) const;

    bool executeGetOperands(GetOperandsOp op, InterpreterState& state) const;
    
    bool executeGetResult(GetResultOp op, InterpreterState& state) const ;

    bool executeGetResults(GetResultsOp op, InterpreterState& state) const ;
    
    bool executeExtractOp(ExtractOp op, InterpreterState& state) const ;

    bool executeGetAttribute(GetAttributeOp op, InterpreterState& state) const;
    
    bool executeRecordMatch(RecordMatchOp op, InterpreterState& state) const;
    
    bool executeApplyRewrite(ApplyRewriteOp op, PatternRewriter& rewriter, InterpreterState& state) const;

    bool executeIsNotNull(IsNotNullOp op, InterpreterState& state) const;

    bool executeAreEqual(AreEqualOp op, InterpreterState& state) const;

    bool executeGetUsers(GetUsersOp op, InterpreterState& state) const;

    bool executeGetDefiningOp(GetDefiningOpOp op, InterpreterState& state) const;

    bool executeForLoop(ForEachOp op, InterpreterState& state) const;
    
    bool executeBlock(Block& block, InterpreterState& state) const;
    
    bool executeRegion(Region& region, InterpreterState& state) const;

    bool executeEraseOp(EraseOp op, PatternRewriter& rewriter, InterpreterState& state) const;
    
    bool loadPDLInterpFile(const std::string& filename);

    bool matchPattern(Operation* rootOp) const;

    //Execute rewrite Ops
    bool rewriteOrExecuteOps(Operation* op, PatternRewriter& rewriter, InterpreterState& state) const;

    void registerNativeRewrites();
public:
    PDLInterpMatcher(MLIRContext *context, const std::string& pdlFile) {
        this->context = context;
        loadPDLInterpFile(pdlFile);
        rewriteState = std::make_unique<RewriteState>();
        registerNativeRewrites();    
    }
    
    
    // Recursively match against IR tree
    bool matchInTree(Operation* op) const;
    

    LogicalResult rewriteModule(PatternRewriter& rewriter) const;

    void dumpPDLModule() {
        if (pdlModule) {
            pdlModule->dump();
        }
    }
};

} // mlir
} // diffusion_graph
#endif