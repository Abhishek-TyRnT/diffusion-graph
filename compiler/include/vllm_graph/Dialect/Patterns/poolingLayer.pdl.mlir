// PDLL pattern to extract constant ops and related custom ops into a new FuncOp

#include "mlir/Dialect/Func/IR/FuncOps.td"
#include "mlir/Dialect/Arith/IR/ArithOps.td"

// Include your custom dialect here
#include "vllm_graph/Dialect/IR/vLLMGraphBase.td"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.td"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.td"



Rewrite CreatePoolingFunc(ops: ValueRange);

// Pattern to identify and extract constant operations along with custom ops
Pattern ExtractPoolingLayer with benefit(1) {
  // Match a function that contains our pattern
  // let root = op<func.func>(
  //   _: SymbolNameAttr,  // function name (don't care)
  //   _: TypeAttr,  // function type
  //   body: region(
  //     // Match operations in the function body
  //     ops: EachOf<[
        // Match constant operations
  let constantOp1 = op<arith.constant>() ;

  let constantOp0 = op<arith.constant>() ;

  let constantZeroTensor = op<vllm_graph.constant.tensor>(); 

  let constantBias = op<vllm_graph.constant.tensor>();

  let constantWeight = op<vllm_graph.constant.tensor>();
    
    // Match custom dialect operations that use constants
    // Replace "custom.op" with your actual custom op
  let layerNormOp = op<vllm_graph.vllm.layer_norm>;

  let IndexSelectOp = op<vllm_graph.vllm.index_select>(layerNormOp, constantOp1, constantZeroTensor);

  let SqueezeOp = op<vllm_graph.vllm.squeeze>(IndexSelectOp, constantOp1) ;

  let TransposeOp = op<vllm_graph.vllm.transpose>(constantBias, constantOp0, constantOp1) ;

  let AddMmOp = op<vllm_graph.vllm.addmm>(constantWeight, SqueezeOp, TransposeOp, constantOp1, constantOp1) ;

  let TanhOp = op<vllm_graph.vllm.tanh>(AddMmOp);

  let OpsToMove = ( constantOp1, 
                                  constantOp0, 
                                  constantZeroTensor, 
                                  constantBias,
                                  constantWeight,
                                  IndexSelectOp,
                                  SqueezeOp,
                                  TransposeOp,
                                  AddMmOp,
                                  TanhOp );

                         
  rewrite TanhOp with {
    CreatePoolingFunc(IndexSelectOp);
    erase IndexSelectOp;
    erase SqueezeOp;
    erase TransposeOp;
    erase AddMmOp;
    erase TanhOp;

  };

  //   //   ]>
  //   // )
  // // );
  
  // // Rewrite logic
  // rewrite IndexSelectOp with {
  //   // Get the module containing the current function
  //   let oldFunc = IndexSelectOp->getParentOfType<func.func>();
    
  //   // Create a new FuncOp to hold the extracted constants
  //   let newFuncName = "get_pooling_output";
  //   let newFuncType = FunctionType::get(
  //     {layerNormResult.getType()},  // no inputs for constant function
  //     {TanhResult.getType()},  // outputs
  //     root.getContext()
  //   );
    
  //   // Build the new function
  //   let newFunc = rewriter.create<func::FuncOp>(
  //     root.getLoc(),
  //     newFuncName,
  //     newFuncType
  //   );
    
  //   // Create a block for the new function
  //   let entryBlock = newFunc.addEntryBlock();
  //   rewriter.setInsertionPointToStart(entryBlock);
    
  //   // Clone the constant operations into the new function
  //   let newconstantOp1 = rewriter.clone(*constantOp1);
  //   let newconstantOp0 = rewriter.clone(*constantOp0);
  //   let newconstantZeroTensor = rewriter.clone(*constantZeroTensor);
  //   let newconstantBias = rewriter.clone(*constantBias);
  //   let newconstantWeight = rewriter.clone(*constantWeight);
  //   let newIndexSelectOp = rewriter.clone(*IndexSelectOp);
  //   let newSqueezeOp = rewriter.clone(*SqueezeOp);
  //   let newTransposeOp = rewriter.clone(*TransposeOp);
  //   let newAddMmOp = rewriter.clone(*AddMmOp);
  //   let newTanhOp = rewriter.clone(*TanhOp);
    
  //   // Update the custom op to use the new constant
  //   newIndexSelectOp.setOperands({ entryBlock.getArgument(0), newconstantOp1.getResult(), newconstantZeroTensor.getResult() });
  //   newSqueezeOp.setOperands({ newIndexSelectOp.getResult(), newconstantOp1.getResult() });
  //   newTransposeOp.setOperands({ newconstantBias.getResult(), newconstantOp0.getResult(), newconstantOp1.getResult() });
  //   newAddMmOp.setOperands({newconstantWeight.getResult(), 
  //                           newSqueezeOp.getResult(), 
  //                           newTransposeOp.getResult(), 
  //                           newconstantOp1.getResult(), 
  //                           newconstantOp1.getResult()});
  //   newTanhOp.setOperand(0, newAddMmOp);
  //   // Add a return statement to the new function
  //   SmallVector<Value> results = {
  //     newTanhOp.getResult()
  //   };
  //   rewriter.create<func::ReturnOp>(root.getLoc(), results);
    
  //   // Insert the new function into the module
  //   module.push_back(newFunc);
    
  //   // Note: Not creating a call from original to new function as requested
  //   // The original ops remain in place
  // };
}

// Alternative pattern using native C++ code for more complex scenarios
// Pattern ExtractConstantsComplex : NativeCodePattern<"ExtractConstantsComplex"> {
//   let benefitScore = 10;
  
//   let matchAndRewriteFunc = [{
//     auto funcOp = dyn_cast<func::FuncOp>(op);
//     if (!funcOp)
//       return failure();
    
//     // Collect all constant ops and their dependent custom ops
//     SmallVector<arith::ConstantOp> constantOps;
//     SmallVector<Operation*> customOps;
//     DenseSet<Operation*> toExtract;
    
//     funcOp.walk([&](Operation *op) {
//       if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
//         constantOps.push_back(constantOp);
//         toExtract.insert(op);
        
//         // Find custom ops that use this constant
//         for (auto user : op->getUsers()) {
//           // Check if user is from custom dialect
//           if (user->getDialect()->getNamespace() == "custom") {
//             customOps.push_back(user);
//             toExtract.insert(user);
//           }
//         }
//       }
//     });
    
//     // If no constants found, pattern doesn't match
//     if (constantOps.empty())
//       return failure();
    
//     // Create new function to hold extracted ops
//     OpBuilder builder(funcOp);
//     auto loc = funcOp.getLoc();
//     auto modulOp = funcOp->getParentOfType<ModuleOp>();
    
//     // Build function type based on extracted ops
//     SmallVector<Type> resultTypes;
//     for (auto *op : toExtract) {
//       for (auto result : op->getResults()) {
//         resultTypes.push_back(result.getType());
//       }
//     }
    
//     auto newFuncType = FunctionType::get(
//       &getContext(),
//       /*inputs=*/{},
//       /*results=*/resultTypes
//     );
    
//     // Create the new function
//     auto newFuncName = (funcOp.getName() + "_extracted_constants").str();
//     auto newFunc = builder.create<func::FuncOp>(
//       loc,
//       newFuncName,
//       newFuncType
//     );
    
//     // Create entry block and builder for new function
//     auto *entryBlock = newFunc.addEntryBlock();
//     OpBuilder::InsertionGuard guard(builder);
//     builder.setInsertionPointToStart(entryBlock);
    
//     // Clone operations into new function
//     IRMapping mapping;
//     SmallVector<Value> returnValues;
    
//     for (auto *op : toExtract) {
//       Operation *cloned = builder.clone(*op, mapping);
//       for (auto result : cloned->getResults()) {
//         returnValues.push_back(result);
//       }
//     }
    
//     // Add return statement
//     builder.create<func::ReturnOp>(loc, returnValues);
    
//     // Insert new function into module
//     builder.setInsertionPoint(funcOp);
//     builder.insert(newFunc);
    
//     return success();
//   }];
// }

// // Constraint helper to check if a value is produced by a constant
// Constraint ConstantUseConstraint(use: Value, def: Value) [{
//   return use == def;
// }]

// // Pattern for matching specific constant patterns with custom dialect ops
// Pattern SpecificConstantPattern {
//   // Match specific constant integer value
//   let constOp = op<arith.constant> {value = attr<"42 : i32">} -> (result: Value);
  
//   // Match custom op that uses this specific constant
//   let customOp = op<custom.compute>(constOp, arg: Value) -> (output: Value);
  
//   // Match the enclosing function
//   let func = op<func.func>(_, _, _, body: Block(contains: [constOp, customOp]));
  
//   rewrite func with {
//     // Create specialized function for this pattern
//     let newFunc = op<func.func>(
//       name = func.getName() + "_const42_pattern",
//       type = FunctionType::get({arg.getType()}, {output.getType()}),
//       body = Block(
//         // Recreate the pattern in new function
//         newConst: op<arith.constant> {value = attr<"42 : i32">},
//         newCustom: op<custom.compute>(newConst, arg),
//         op<func.return>(newCustom)
//       )
//     );
    
//     // Add to module
//     func.getParentOp().push_back(newFunc);
//   };
// }