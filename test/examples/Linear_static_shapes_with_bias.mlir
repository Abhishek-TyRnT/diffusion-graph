module {
  func.func @main(%arg0: !torch.vtensor<[1,224,10],f32>) -> !torch.vtensor<[1,224,5],f32> {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_10_torch.float32> : tensor<5x10xf32>) : !torch.vtensor<[5,10],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>) : !torch.vtensor<[5],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.aten.transpose.int %0, %int0, %int1 : !torch.vtensor<[5,10],f32>, !torch.int, !torch.int -> !torch.vtensor<[10,5],f32>
    %3 = torch.aten.matmul %arg0, %2 : !torch.vtensor<[1,224,10],f32>, !torch.vtensor<[10,5],f32> -> !torch.vtensor<[1,224,5],f32>
    %4 = torch.aten.add.Tensor %3, %1, %int1 : !torch.vtensor<[1,224,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[1,224,5],f32>
    return %4 : !torch.vtensor<[1,224,5],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_5_10_torch.float32: "0x040000009F8B923E0E132BBD38D3F1BDB30ECDBD6BC827BE622053BB0EDC70BE4DBC953E7250453EBEA0393EF6EA2DBE800E903DAEF293BDCE0690BE117A1EBEE90359BE479E7FBE39C2A13E721CFCBD2427EC3DDC029E3E5F5A073E1C93A5BDC1841A3E84EEC1BDFC7523BE67E7163EB51FE23CD7D7893E0AFFAEBBFB032CBC441788BDE5F1CE3DABA7053E7F9694BE06FE62BEDE1B853E5F38FF3CF808033C04A986BE5D6FD03D438429BE1BCF64BE14DC55BE01CE1E3DEF4A663EB2CF4DBD3E096A3E0D98873E1E89C33D",
      torch_tensor_5_torch.float32: "0x0400000000ED79BE4F89C2BD7EC08A3E36DB80BECD6E8E3E"
    }
  }
#-}
