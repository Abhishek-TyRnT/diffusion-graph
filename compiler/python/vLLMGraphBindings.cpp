#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For STL containers
#include <pybind11/complex.h>
#include "Interface.hpp"
#include "GraphConvertor/GraphWriter.hpp"
#include <mlir/IR/Dialect.h>
#include "mlir/IR/BuiltinDialect.h"
#include <unordered_map>
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include <any>
#include <iostream>

#ifdef MLIR_PYTHON_PACKAGE_PREFIX
#undef MLIR_PYTHON_PACKAGE_PREFIX
#endif
#define MLIR_PYTHON_PACKAGE_PREFIX torch_mlir._mlir_libs._mlir.
#define MLIR_PYTHON_CAPSULE_MODULE MAKE_MLIR_PYTHON_QUALNAME("ir.Module._CAPIPtr")
// #define MLIMLIR_PYTHON_CAPI_PTR_ATTR 

namespace py = pybind11;
using namespace mlir::vllm_graph;

static MlirModule castPythonObjectToMlirModule(py::object& moduleObj) {
  assert(!moduleObj.is_none() && "module cannot be None");
  auto moduleCapsule = moduleObj.attr("_CAPIPtr");
  const char* capsuleName = PyCapsule_GetName(moduleCapsule.ptr());
  std::cout << MLIR_PYTHON_CAPSULE_MODULE << std::endl;
  std::cout << "CapsuleName :- " << capsuleName << std::endl;
  void *ptr = PyCapsule_GetPointer(moduleCapsule.ptr(), capsuleName);
  MlirModule module = {ptr};

  if (mlirModuleIsNull(module)) {
    // An error will have already been set by the above.
    throw py::error_already_set();
  }

  return module;
}

static MlirContext castPythonObjectToMlirContext(py::object& contextObj) {
  assert(!contextObj.is_none() && "context cannot be None");
  auto contextCapsule = contextObj.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
  void *ptr = PyCapsule_GetPointer(contextCapsule.ptr(), MLIR_PYTHON_CAPSULE_CONTEXT);
  MlirContext context = {ptr};
  if (mlirContextIsNull(context)) {
    // An error will have already been set by the above.
    throw py::error_already_set();
  }
  return context;
}

class vLLMGraph : public vLLMGraphBase{

private:
    GraphWriter convertor;
    // mlir::OwningOpRef<mlir::ModuleOp> process_mlir_module(MlirModule module) {
    // // Unwrap to C++ mlir::ModuleOp
    // mlir::ModuleOp modOp = llvm::cast<mlir::ModuleOp>(
    //     mlir::OpInterface<mlir::Operation>::getInterfaceFor(
    //         mlir::unwrap(module)
    //     )
    // );
    // // -- OR the simpler direct cast: --
    // // mlir::Operation* op = mlir::unwrap(module);
    // // mlir::ModuleOp modOp = llvm::cast<mlir::ModuleOp>(op);

    // mlir::OwningOpRef<mlir::ModuleOp> ModuleRef = mlir::OwningOpRef<mlir::ModuleOp>(modOp);
    // // TODO: add your real processing here
    // return ModuleRef;
    // }

public:

vLLMGraph(std::string weightsPath) : convertor(weightsPath){}

std::unordered_map<std::string, std::variant<SubGraphMap, std::string>> compile(std::string &ir_path){
    mlir::OwningOpRef<mlir::ModuleOp> moduleRef = parseFromFile(ir_path);
    convert(moduleRef);
    convertor.build(moduleRef, dialectResourcesMap);
    convertor.closeFile();    
    return convertor.getGraph();
}

// std::unordered_map<std::string, SubGraphMap> compile(py::object IR){
//     std::cout << "Starting IR Loading!" << std::endl;
//     py::object capsule = IR.attr("_CAPIPtr");

//             // Unwrap the PyCapsule -> void* -> MlirModule
//             // The capsule name used by MLIR Python bindings is "mlir.ir.Module._CAPIPtr"
//     void* ptr = PyCapsule_GetPointer(
//         capsule.ptr(),
//         "torch_mlir._mlir_libs._mlir.ir.Module._CAPIPtr"   // must match MLIR's registered capsule name
//     );
//     if (!ptr) {
//         throw std::runtime_error("Failed to get pointer from capsule");
//     }
//     MlirModule cModule = {ptr};  // wrap raw pointer into C API struct
//     mlir::OwningOpRef<mlir::ModuleOp> moduleRef = process_mlir_module(cModule);

//     std::cout << "Loaded IR!" << std::endl;
//     convert(moduleRef);
//     std::cout << "Completed IR Conversion!" << std::endl;
//     convertor.build(moduleRef, dialectResourcesMap);
//     std::cout << "Completed Graph Building!" << std::endl;
//     convertor.closeFile();
//     std::cout << "Completed Graph Closing!" << std::endl;
    
//     return convertor.getGraph();
// }

void compile(py::object moduleObj){
    // MlirContext contextC = castPythonObjectToMlirContext(contextObj);
    // mlir::MLIRContext *context = unwrap(contextC);
    // context->loadAllAvailableDialects();
    // for (auto dialect : context->getLoadedDialects()) {
    //     llvm::outs() << dialect->getNamespace() << "\n";
    // }
    MlirModule moculeC = castPythonObjectToMlirModule(moduleObj);
    mlir::ModuleOp module = unwrap(moculeC);
    mlir::MLIRContext *context = module.getContext();
    context->loadAllAvailableDialects();
    std::cout << __LINE__ << std::endl;
  
  // Check if the type's context matches the module's context
    // module.walk([&](Operation *op) {
    //     for (auto namedAttr : op->getAttrs()) {
    //         if (auto resourceAttr = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(namedAttr.getValue())) {
    //             // Get the resource handle and blob
    //             auto handle = resourceAttr.getRawHandle();
    //             StringRef key = handle.getKey();
                
    //             // Get the blob data
    //             if (auto *blob = handle.getBlob()) {
    //                 llvm::ArrayRef<char> data = blob->getData();
    //                 llvm::outs() << key << ": " << data[0] << "\n";
    //             }
    //         }
    //     }
    // });
    // auto *manager = context->getOrCreateResourceBlobManager();

    std::cout << __LINE__ << std::endl;
    mlir::OwningOpRef<mlir::ModuleOp> moduleRef(module);
    std::cout << __LINE__ << std::endl;
    convert(moduleRef);
    std::cout << __LINE__ << std::endl;
    convertor.build(moduleRef, dialectResourcesMap);
    std::cout << __LINE__ << std::endl;
    convertor.closeFile();
    std::cout << __LINE__ << std::endl;
}


};

PYBIND11_MODULE(graph_compiler, m){

    py::class_<vLLMGraph>(m, "vLLMGraph")
        .def(py::init<std::string& >())
        .def("compile", py::overload_cast<std::string&>(&vLLMGraph::compile))
        .def("compile", py::overload_cast<py::object>(&vLLMGraph::compile));
}