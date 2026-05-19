
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "InitAll.hpp"

using namespace mlir;

int main(int argc, char **argv) {
    mlir::diffusion_graph::registerAllPasses();

    DialectRegistry registry;
    mlir::diffusion_graph::registerAllDialects(registry);
    return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
    return 0;

}