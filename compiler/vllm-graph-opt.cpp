
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "InitAll.h"

using namespace mlir;

int main(int argc, char **argv) {
    mlir::vllm_graph::registerAllPasses();

    DialectRegistry registry;
    mlir::vllm_graph::registerAllDialects(registry);
    return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}