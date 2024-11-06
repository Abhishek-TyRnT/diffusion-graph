#include <pybind11/pybind11.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/Pass.h>
#include "InitAllc.h"
#include "InitAll.hpp"

namespace py = pybind11;
using namespace mlir::vllm_graph;

PYBIND11_MODULE(graph_compiler, m){

    m.doc() = "Python bindings for vllm graph compiler";

    m.def("registerDialect", &registervLLMGraphDialect);
    m.def("registerPasses", &registervLLMGraphPasses);
}