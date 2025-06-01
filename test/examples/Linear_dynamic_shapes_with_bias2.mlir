module attributes {torch.debug_module_name = "Linear"} {
  func.func @forward(%arg0: !torch.vtensor<[?,?,3],f32>) -> !torch.vtensor<[?,?,10],f32> {
    %0 = torch.vtensor.literal(dense<[-0.423338294, 0.249682099, 0.0942039564, 0.268863916, -0.205923319, -0.430605441, 0.323684126, 0.142657667, 0.198047683, 0.280244976]> : tensor<10xf32>) : !torch.vtensor<[10],f32>
    %1 = torch.vtensor.literal(dense<[[-1.64080033E-4, 0.412616789, 0.127303869], [-0.378704756, 0.345147431, 0.247985423], [-0.331900984, -0.512360692, 0.0219810791], [0.396312654, 0.380702972, -0.30433479], [0.199749053, 0.253878057, 0.436269104], [0.423063338, -0.436686546, 0.200477704], [0.0357305594, -0.291539222, -0.00450552441], [0.237356558, -0.313713431, -0.415559649], [-0.277372122, 0.256510228, -0.323375851], [-0.24392657, -0.32959637, -0.0513026752]]> : tensor<10x3xf32>) : !torch.vtensor<[10,3],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[10,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,10],f32>
    %3 = torch.aten.matmul %arg0, %2 : !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[3,10],f32> -> !torch.vtensor<[?,?,10],f32>
    %4 = torch.aten.add.Tensor %3, %0, %int1 : !torch.vtensor<[?,?,10],f32>, !torch.vtensor<[10],f32>, !torch.int -> !torch.vtensor<[?,?,10],f32>
    return %4 : !torch.vtensor<[?,?,10],f32>
  }
}
