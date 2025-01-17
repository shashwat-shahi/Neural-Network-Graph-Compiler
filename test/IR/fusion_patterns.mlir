// RUN: nnc-opt %s --nn-operator-fusion | FileCheck %s

// Test Conv+ReLU fusion
func.func @test_conv_relu_fusion(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> {
  // CHECK: nn.fused_conv_relu
  // CHECK-NOT: nn.conv
  // CHECK-NOT: nn.relu
  %0 = nn.conv %arg0, %arg1 {strides = [2, 2], padding = [3, 3]} : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32> -> tensor<1x64x112x112xf32>
  %1 = nn.relu %0 : tensor<1x64x112x112xf32> -> tensor<1x64x112x112xf32>
  return %1 : tensor<1x64x112x112xf32>
}

// Test MatMul+Add pattern (demonstrates fusion capability)
func.func @test_matmul_add(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %0 = nn.matmul %arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32> -> tensor<128x512xf32>
  %1 = nn.add %0, %arg2 : tensor<128x512xf32>, tensor<128x512xf32> -> tensor<128x512xf32>
  return %1 : tensor<128x512xf32>
}