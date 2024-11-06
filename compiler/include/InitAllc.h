#ifndef INIT_ALL_C_API_H_
#define INIT_ALL_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <mlir-c/IR.h>

void registervLLMGraphDialect(MlirContext &contextc);

void registervLLMGraphPasses();

#ifdef __cplusplus
}
#endif

#endif

