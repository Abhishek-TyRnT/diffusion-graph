#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For STL containers
#include <pybind11/complex.h>
#include "Interface.hpp"
#include "GraphConvertor/GraphWriter.hpp"
#include <mlir/IR/Dialect.h>
// #include <mlir/Pass/Pass.h>
#include "mlir/IR/BuiltinDialect.h"
#include <unordered_map>
#include <any>
// #include "InitAllc.h"
// #include "InitAll.hpp"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"
#include <iostream>

namespace py = pybind11;
using namespace mlir::vllm_graph;

// using NestedValueType = std::variant<std::string, std::vector<int64_t>>;
// using ValueType = std::variant<std::string, 
//                     std::unordered_map<std::string, NestedValueType>>;


static py::object getMlirIrClass(const char* className) {
  return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr(className);
}

static py::object castMlirModuleToPythonObject(MlirModule *module) {
  auto moduleClass = getMlirIrClass("Module");
  auto moduleCapsule =
      py::reinterpret_steal<py::object>(mlirPythonModuleToCapsule(*module));
  py::object obj = moduleClass.attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(moduleCapsule);
  return obj;
}

// static MlirModule castPythonObjectToMlirContext(py::object& moduleobj) {
//   assert(!moduleobj.is_none() && "context cannot be None");
//   auto contextCapsule = moduleobj.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
  
//   MlirModule context = mlirPythonCapsuleToModule(contextCapsule.ptr());
//   llvm::outs() << *unwrap(context) << "\n";
//   if (mlirModuleIsNull(context)) {
//     // An error will have already been set by the above.
//     throw py::error_already_set();
//   }
//   return context;
// }

class vLLMGraph : public vLLMGraphBase{

private:
    GraphWriter convertor;

public:
std::unordered_map<std::string, ValueType> compile(std::string IRFile){
    mlir::OwningOpRef<mlir::ModuleOp> moduleRef = convert(IRFile);
    convertor.build(moduleRef);
    
    return convertor.getGraph();
    // module = moduleRef.get();
    // MlirModule moduleC = wrap(module);
    // moduleC_ptr = &moduleC;
    // moduleObj = castMlirModuleToPythonObject(moduleC);
    //castPythonObjectToMlirContext(moduleObj);
}

};

PYBIND11_MODULE(graph_compiler, m){

    py::class_<vLLMGraph>(m, "vLLMGraph")
        .def(py::init())
        .def("convert", &vLLMGraph::compile);
    // m.doc() = "Python bindings for vllm graph compiler";

    // m.def("registerDialect", &registervLLMGraphDialect);
    // m.def("registerPasses", &registervLLMGraphPasses);
}