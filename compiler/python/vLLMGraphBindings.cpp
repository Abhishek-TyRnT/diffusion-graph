#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For STL containers
#include <pybind11/complex.h>
#include "Interface.hpp"
#include "GraphConvertor/GraphWriter.hpp"
#include <mlir/IR/Dialect.h>
#include "mlir/IR/BuiltinDialect.h"
#include <unordered_map>
#include <any>
#include <iostream>

namespace py = pybind11;
using namespace mlir::vllm_graph;

class vLLMGraph : public vLLMGraphBase{

private:
    GraphWriter convertor;

public:

vLLMGraph(std::string weightsPath) : convertor(weightsPath){}

std::unordered_map<std::string, SubGraphMap> compile(std::string IR){
    mlir::OwningOpRef<mlir::ModuleOp> moduleRef = parse(IR);
    convert(moduleRef);
    convertor.build(moduleRef);
    convertor.closeFile();
    
    return convertor.getGraph();
}

};

PYBIND11_MODULE(graph_compiler, m){

    py::class_<vLLMGraph>(m, "vLLMGraph")
        .def(py::init<std::string& >())
        .def("compile", &vLLMGraph::compile);
}