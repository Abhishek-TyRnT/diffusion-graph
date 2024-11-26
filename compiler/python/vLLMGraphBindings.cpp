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

class vLLMGraph : public vLLMGraphBase{

private:
    GraphWriter convertor;

public:
std::unordered_map<std::string, ValueType> compile(std::string IRFile){
    mlir::OwningOpRef<mlir::ModuleOp> moduleRef = convert(IRFile);
    convertor.build(moduleRef);
    
    return convertor.getGraph();
}

};

PYBIND11_MODULE(graph_compiler, m){

    py::class_<vLLMGraph>(m, "vLLMGraph")
        .def(py::init())
        .def("convert", &vLLMGraph::compile);
}