module attributes {torch.debug_module_name = "Softmax"} {
  func.func @forward(%arg0: !torch.vtensor<[5,3,224,224],f32>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[5,3,224,224],f32> {
    %int1 = torch.constant.int 1
    %none = torch.constant.none
    %0 = torch.aten.softmax.int %arg0, %int1, %none : !torch.vtensor<[5,3,224,224],f32>, !torch.int, !torch.none -> !torch.vtensor<[5,3,224,224],f32>
    return %0 : !torch.vtensor<[5,3,224,224],f32>
  }
}
