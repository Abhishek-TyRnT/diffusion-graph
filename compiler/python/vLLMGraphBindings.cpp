#include <pybind11/pybind11.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/Pass.h>
#include "mlir/IR/BuiltinDialect.h"
#include "InitAllc.h"
#include "InitAll.hpp"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"
#include <iostream>

namespace py = pybind11;
using namespace mlir::vllm_graph;

static py::object getMlirIrClass(const char* className) {
  return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr(className);
}

static MlirContext castPythonObjectToMlirContext(py::object& contextObj) {
  assert(!contextObj.is_none() && "context cannot be None");
  auto contextCapsule = contextObj.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
  MlirContext context = mlirPythonCapsuleToContext(contextCapsule.ptr());
  if (mlirContextIsNull(context)) {
    // An error will have already been set by the above.
    throw py::error_already_set();
  }
  return context;
}

static py::object castMlirContextToPythonObject(MlirContext& context) {
  auto contextClass = getMlirIrClass("Context");
  auto contextCapsule = py::reinterpret_steal<py::object>(mlirPythonContextToCapsule(context));
  return contextClass.attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(contextCapsule);
}

py::object registervLLMGraphDialect(){
    mlir::DialectRegistry registry;
    mlir::vllm_graph::registerAllDialects(registry);
    registry.insert<mlir::BuiltinDialect>();
    MlirDialectRegistry registryc;
    registryc.ptr = &registry;
    // if(!unwrap(context))
    // {
    //   std::cout << "context is null " << std::endl;
    // }
    MlirContext context = mlirContextCreateWithRegistry(registryc, true);
    return castMlirContextToPythonObject(context);
}

void registervLLMGraphPasses(){
    mlir::vllm_graph::registerAllPasses();
}

PYBIND11_MODULE(graph_compiler, m){

    m.doc() = "Python bindings for vllm graph compiler";

    m.def("registerDialect", &registervLLMGraphDialect);
    m.def("registerPasses", &registervLLMGraphPasses);
}