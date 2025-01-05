module attributes {torch.debug_module_name = "NewGELUActivation"} {
  func.func @forward(%arg0: !torch.vtensor<[3,256,1024],f32>) -> !torch.vtensor<[3,256,1024],f32> {
    %float3.000000e00 = torch.constant.float 3.000000e+00
    %float4.471500e-02 = torch.constant.float 4.471500e-02
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %float5.000000e-01 = torch.constant.float 5.000000e-01
    %float7.978850e-01 = torch.constant.float 0.79788456080286541
    %int1 = torch.constant.int 1
    %0 = torch.aten.mul.Scalar %arg0, %float5.000000e-01 : !torch.vtensor<[3,256,1024],f32>, !torch.float -> !torch.vtensor<[3,256,1024],f32>
    %1 = torch.aten.pow.Tensor_Scalar %arg0, %float3.000000e00 : !torch.vtensor<[3,256,1024],f32>, !torch.float -> !torch.vtensor<[3,256,1024],f32>
    %2 = torch.aten.mul.Scalar %1, %float4.471500e-02 : !torch.vtensor<[3,256,1024],f32>, !torch.float -> !torch.vtensor<[3,256,1024],f32>
    %3 = torch.aten.add.Tensor %arg0, %2, %int1 : !torch.vtensor<[3,256,1024],f32>, !torch.vtensor<[3,256,1024],f32>, !torch.int -> !torch.vtensor<[3,256,1024],f32>
    %4 = torch.aten.mul.Scalar %3, %float7.978850e-01 : !torch.vtensor<[3,256,1024],f32>, !torch.float -> !torch.vtensor<[3,256,1024],f32>
    %5 = torch.aten.tanh %4 : !torch.vtensor<[3,256,1024],f32> -> !torch.vtensor<[3,256,1024],f32>
    %6 = torch.aten.add.Scalar %5, %float1.000000e00, %int1 : !torch.vtensor<[3,256,1024],f32>, !torch.float, !torch.int -> !torch.vtensor<[3,256,1024],f32>
    %7 = torch.aten.mul.Tensor %0, %6 : !torch.vtensor<[3,256,1024],f32>, !torch.vtensor<[3,256,1024],f32> -> !torch.vtensor<[3,256,1024],f32>
    return %7 : !torch.vtensor<[3,256,1024],f32>
  }
}
