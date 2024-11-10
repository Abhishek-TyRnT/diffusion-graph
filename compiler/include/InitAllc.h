#ifndef INIT_ALL_C_API_H_
#define INIT_ALL_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <mlir-c/IR.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object registervLLMGraphDialect();

void registervLLMGraphPasses();

#ifdef __cplusplus
}
#endif

#endif

