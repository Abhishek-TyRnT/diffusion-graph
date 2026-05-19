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
using namespace mlir::diffusion_graph;

class diffusionGraph : public diffusionGraphBase {

private:
    GraphWriter convertor;

public:

diffusionGraph(std::string weightsPath) : convertor(weightsPath){}

std::unordered_map<std::string, std::variant<SubGraphMap, std::string>> compile(std::string &ir_path){
    mlir::OwningOpRef<mlir::ModuleOp> moduleRef = parseFromFile(ir_path);
    convert(moduleRef);
    convertor.build(moduleRef, dialectResourcesMap);
    convertor.closeFile();    
    return convertor.getGraph();
}

};

PYBIND11_MODULE(graph_compiler, m){

    py::class_<diffusionGraph>(m, "diffusionGraph")
        .def(py::init<std::string& >())
        .def("compile", py::overload_cast<std::string&>(&diffusionGraph::compile));
}