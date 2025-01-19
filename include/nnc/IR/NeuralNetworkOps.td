//===- NeuralNetworkOps.td - Neural Network dialect ops --*- tablegen -*-===//
//
// Neural Network Graph Compiler - MLIR Neural Network Operations
//
//===----------------------------------------------------------------------===//

#ifndef NEURAL_NETWORK_OPS
#define NEURAL_NETWORK_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NeuralNetwork_Dialect : Dialect {
  let name = "nn";
  let summary = "Neural Network operations dialect";
  let description = [{
    This dialect contains operations for neural network graph compilation.
    It supports tensor operations, activation functions, and optimization passes.
  }];
  let cppNamespace = "::nnc::nn";
}

// Base class for neural network operations
class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NeuralNetwork_Dialect, mnemonic, traits>;

// Tensor type for neural network operations
def TensorType : Type<CPred<"$_self.isa<mlir::TensorType>()">, "tensor">;

//===----------------------------------------------------------------------===//
// Neural Network Operations
//===----------------------------------------------------------------------===//

def NN_ConvOp : NN_Op<"conv", [Pure]> {
  let summary = "Convolution operation";
  let description = [{
    Performs 2D convolution operation with optional padding and stride.
    Supports operator fusion optimization.
  }];
  
  let arguments = (ins TensorType:$input, TensorType:$filter, 
                       I64ArrayAttr:$strides, I64ArrayAttr:$padding);
  let results = (outs TensorType:$output);
  
  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($output)
  }];
}

def NN_MatMulOp : NN_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication with automatic vectorization support.
  }];
  
  let arguments = (ins TensorType:$lhs, TensorType:$rhs);
  let results = (outs TensorType:$output);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def NN_ReluOp : NN_Op<"relu", [Pure]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies ReLU activation function: max(0, x).
    Optimized for operator fusion.
  }];
  
  let arguments = (ins TensorType:$input);
  let results = (outs TensorType:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def NN_AddOp : NN_Op<"add", [Pure, Commutative]> {
  let summary = "Element-wise addition";
  let description = [{
    Performs element-wise addition with broadcasting support.
  }];
  
  let arguments = (ins TensorType:$lhs, TensorType:$rhs);
  let results = (outs TensorType:$output);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def NN_FusedConvReluOp : NN_Op<"fused_conv_relu", [Pure]> {
  let summary = "Fused convolution and ReLU operation";
  let description = [{
    Fused convolution followed by ReLU activation for improved performance.
    Result of operator fusion optimization.
  }];
  
  let arguments = (ins TensorType:$input, TensorType:$filter,
                       I64ArrayAttr:$strides, I64ArrayAttr:$padding);
  let results = (outs TensorType:$output);
  
  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($output)
  }];
}

#endif // NEURAL_NETWORK_OPS