module attributes {torch.debug_module_name = "TransposeModel"} {
  func.func @forward(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg0, %int1, %int2 : !torch.vtensor<[?,?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,?],f32>
    return %0 : !torch.vtensor<[?,?,?],f32>
  }
}
