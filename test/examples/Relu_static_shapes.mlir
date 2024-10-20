module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: !torch.vtensor<[1, 10, 3],f32>) -> !torch.vtensor<[1, 10, 3],f32> {
    %0 = torch.aten.relu %arg0 : !torch.vtensor<[1, 10, 3],f32> -> !torch.vtensor<[1, 10, 3],f32>
    return %0 : !torch.vtensor<[1, 10, 3],f32>
  }
}
