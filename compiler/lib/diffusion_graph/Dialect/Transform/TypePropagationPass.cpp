// TypePropagation.cpp
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Debug.h"
#include "diffusion_graph/Dialect/IR/DiffusionGraphDialect.hpp"
#include "diffusion_graph/Dialect/IR/DiffusionGraphOps.hpp"
#include "diffusion_graph/Dialect/IR/DiffusionGraphTypes.hpp"
#include "diffusion_graph/Dialect/Transform/Passes.hpp"

#include "PassDetail.hpp"

using namespace mlir;
using namespace diffusion_graph;

namespace {

/// Ops whose types are part of a fixed external contract (function
/// signature, call ABI) and must never be silently rewritten. Extend
/// for any other boundary ops in your pipeline (e.g. a serialization
/// boundary between compiled stages).
static bool isTypeFixed(Operation *op) {
  return isa<func::ReturnOp, func::CallOp>(op);
}

/// Recompute and apply result types for a single op via
/// InferTypeOpInterface. Returns true if anything changed.
static bool propagateTypes(Operation *op, RewriterBase &rewriter) {
  auto inferIface = dyn_cast<InferTypeOpInterface>(op);
  if (!inferIface)
    return false; // no interface: nothing this pass can infer
  
  SmallVector<Type, 4> inferred;
  if (failed(inferIface.inferReturnTypes(
          op->getContext(), op->getLoc(), op->getOperands(),
          op->getAttrDictionary(), op->getPropertiesStorage(),
          op->getRegions(), inferred))) {
    op->emitWarning() << "type-propagation: type inference failed, "
                          "leaving op unchanged";
    return false;
  }

  if (inferred.size() != op->getNumResults()) {
    op->emitWarning() << "type-propagation: inferred result count ("
                       << inferred.size() << ") does not match op result "
                          "count (" << op->getNumResults() << "), skipping";
    return false;
  }


  bool changed = false;
  for (auto [result, newType] : llvm::zip(op->getResults(), inferred)) {

    if (result.getType() == newType)
      continue;
    // LLVM_DEBUG(llvm::dbgs() << "  " << result << " : "
    //                         << result.getType() << " -> " << newType << "\n");
    Value res = result;
    Type ty = newType;
    rewriter.modifyOpInPlace(op, [&] { res.setType(ty); });    
    changed = true;
  }
  return changed;
}

/// Insert a cast at `use` if the operand's current type doesn't match
/// `expectedType`. Customize diffusion_graph::CastOp to whatever cast op
/// your dialect defines (must be able to convert between all dtype
/// pairs you expect to see at boundaries).
static void materializeCastIfNeeded(OpOperand &use, Type expectedType,
                                     RewriterBase &rewriter) {
  Value operand = use.get();
  if (operand.getType() == expectedType)
    return;

  rewriter.setInsertionPoint(use.getOwner());
  Value cast = rewriter.create<diffusion_graph::CastOp>(
      use.getOwner()->getLoc(), expectedType, operand);
  rewriter.modifyOpInPlace(use.getOwner(), [&] { use.set(cast); });
}

/// Runs propagateTypes to a fixed point over a single region's body.
/// Needed for regions with loop-carried values (scf.for/scf.while),
/// where a straight-line forward walk isn't sufficient because the
/// terminator's operand types feed back into the block argument types.
static void propagateThroughRegionToFixedPoint(Region &region,
                                                RewriterBase &rewriter,
                                                unsigned maxIterations = 8) {
  bool changed = true;
  unsigned iter = 0;
  while (changed && iter++ < maxIterations) {
    changed = false;
    for (Block &block : region) {
      for (Operation &op : block) {
        if (isTypeFixed(&op))
          continue;

        // Recurse into nested regions first so block arguments there
        // are settled before we infer types of ops that might depend
        // on them structurally (rare, but cheap to be consistent).
        for (Region &nested : op.getRegions())
          propagateThroughRegionToFixedPoint(nested, rewriter, maxIterations);

        if(auto cast_op = dyn_cast<diffusion_graph::CastOp>(op)){
            Value input = cast_op.getOperand();
            Value result = cast_op.getResult();
            result.replaceAllUsesWith(input);
        }
        else if (propagateTypes(&op, rewriter))
          changed = true;
      }

      // If this block is a loop body, its terminator's operand types
      // must match the block arguments of the loop for the next
      // iteration's operands. Sync them explicitly rather than relying
      // on another walk to notice.
    //   if (auto forOp = dyn_cast_or_null<scf::ForOp>(region.getParentOp())) {
    //     Operation *terminator = block.getTerminator();
    //     for (auto [arg, yieldedOperand] :
    //          llvm::zip(forOp.getRegionIterArgs(), terminator->getOperands())) {
    //       if (arg.getType() != yieldedOperand.getType()) {
    //         rewriter.modifyOpInPlace(forOp, [&] {
    //           arg.setType(yieldedOperand.getType());
    //         });
    //         changed = true;
    //       }
    //     }
    //   }
    }
  }

  if (iter >= maxIterations && changed) {
    region.getParentOp()->emitWarning()
        << "type-propagation: did not converge after " << maxIterations
        << " iterations in region body";
  }
}
} //namespace

namespace {
struct TypePropagationPass
    : public TypePropagationPassBase<TypePropagationPass> {

  StringRef getArgument() const final { return "diffusion_graph-type-propagation"; }
  StringRef getDescription() const final {
    return "Propagate dtype changes through the def-use chain";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Pass 1: propagate inferred types through the function body,
    // including fixed-point handling for loop-carried values.
    propagateThroughRegionToFixedPoint(func.getBody(), rewriter);

    // Pass 2: at fixed-contract boundaries, insert casts wherever an
    // operand's type has drifted from what the boundary expects.
    func.walk([&](Operation *op) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        FunctionType fnType = func.getFunctionType();
        for (auto [idx, use] : llvm::enumerate(returnOp->getOpOperands())) {
          Type expected = fnType.getResult(idx);
          materializeCastIfNeeded(use, expected, rewriter);
        }
      } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            callOp, callOp.getCalleeAttr());
        if (!callee)
          return;
        FunctionType fnType = callee.getFunctionType();
        for (auto [idx, use] : llvm::enumerate(callOp->getOpOperands())) {
          Type expected = fnType.getInput(idx);
          materializeCastIfNeeded(use, expected, rewriter);
        }
      }
    });

  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::diffusion_graph::createTypePropagationPass() {
  return std::make_unique<TypePropagationPass>();
}