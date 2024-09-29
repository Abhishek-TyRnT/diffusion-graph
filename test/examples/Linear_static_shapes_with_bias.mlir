module attributes {torch.debug_module_name = "Linear"} {
  func.func @forward(%arg0: !torch.vtensor<[1,224,3],f32>) -> !torch.vtensor<[1,224,10],f32> {
    %0 = torch.vtensor.literal(dense<[0.0968343988, -0.234596521, -0.0674899518, -0.134468123, -0.12045408, 0.0939144119, 0.514493942, -0.172508255, -0.507236123, -0.410458356]> : tensor<10xf32>) : !torch.vtensor<[10],f32>
    %1 = torch.vtensor.literal(dense<[[0.463694841, 0.277654111, 0.200662643], [-0.193884507, 0.26993072, 0.210507095], [-3.728550e-01, 0.251526624, -0.331323892], [-0.11150153, 0.128870547, -0.0866480842], [-0.505655468, -0.244382948, 0.283765048], [-0.404812604, -0.258208483, -0.285836279], [0.316520751, -0.395455033, 0.429565072], [0.0770826489, -0.0881418735, -0.0923649445], [0.0546372719, 5.417150e-01, 0.557212591], [-0.404822946, 0.194457605, 0.181005597]]> : tensor<10x3xf32>) : !torch.vtensor<[10,3],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[10,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,10],f32>
    %3 = torch.aten.matmul %arg0, %2 : !torch.vtensor<[1,224,3],f32>, !torch.vtensor<[3,10],f32> -> !torch.vtensor<[1,224,10],f32>
    %4 = torch.aten.add.Tensor %3, %0, %int1 : !torch.vtensor<[1,224,10],f32>, !torch.vtensor<[10],f32>, !torch.int -> !torch.vtensor<[1,224,10],f32>
    return %4 : !torch.vtensor<[1,224,10],f32>
  }
}
