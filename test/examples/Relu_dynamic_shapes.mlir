module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
    %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?,?,?],f32> -> !torch.vtensor<[?,?,?,?],f32>
    return %0 : !torch.vtensor<[?,?,?,?],f32>
  }
}
