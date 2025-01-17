// RUN: nnc-opt %s --nn-memory-layout | FileCheck %s

// Test memory layout optimization for convolution
func.func @test_conv_memory_layout(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> {
  // CHECK: nn.conv{{.*}}memory_layout = "NCHW_optimized"{{.*}}vectorization = true
  %0 = nn.conv %arg0, %arg1 {strides = [2, 2], padding = [3, 3]} : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32> -> tensor<1x64x112x112xf32>
  return %0 : tensor<1x64x112x112xf32>
}

// Test memory layout optimization for matrix multiplication
func.func @test_matmul_memory_layout(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  // CHECK: nn.matmul{{.*}}memory_layout = "blocked"{{.*}}vectorization = true
  %0 = nn.matmul %arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32> -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}