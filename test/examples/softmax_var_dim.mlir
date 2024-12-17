module attributes {torch.debug_module_name = "Softmax"} {
  func.func @forward(%arg0: !torch.vtensor<[5,3,224,224],f32>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[5,3,224,224],f32> {
    %none = torch.constant.none
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %true = torch.constant.bool true
    %0 = torch.aten.IntImplicit %arg1 : !torch.vtensor<[],si32> -> !torch.int
    %values, %indices = torch.aten.max.dim %arg0, %0, %true : !torch.vtensor<[5,3,224,224],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],si64>
    %1 = torch.aten.sub.Tensor %arg0, %values, %float1.000000e00 : !torch.vtensor<[5,3,224,224],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.float -> !torch.vtensor<[5,3,224,224],f32>
    %2 = torch.aten.exp %1 : !torch.vtensor<[5,3,224,224],f32> -> !torch.vtensor<[5,3,224,224],f32>
    %3 = torch.prim.ListConstruct %0 : (!torch.int) -> !torch.list<int>
    %4 = torch.aten.sum.dim_IntList %2, %3, %true, %none : !torch.vtensor<[5,3,224,224],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
    %5 = torch.aten.div.Tensor %2, %4 : !torch.vtensor<[5,3,224,224],f32>, !torch.vtensor<[?,?,?,?],f32> -> !torch.vtensor<[5,3,224,224],f32>
    return %5 : !torch.vtensor<[5,3,224,224],f32>
  }
}
