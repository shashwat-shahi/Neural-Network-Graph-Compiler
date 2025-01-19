//===- NeuralNetworkDialect.cpp - Neural Network dialect ------*- C++ -*-===//
//
// Neural Network Graph Compiler
//
//===----------------------------------------------------------------------===//

#include "nnc/IR/NeuralNetworkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace nnc::nn;

#include "nnc/IR/NeuralNetworkOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Neural Network dialect
//===----------------------------------------------------------------------===//

NeuralNetworkDialect::NeuralNetworkDialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context, TypeID::get<NeuralNetworkDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "nnc/IR/NeuralNetworkOps.cpp.inc"
      >();
}

mlir::Type NeuralNetworkDialect::parseType(mlir::DialectAsmParser &parser) const {
  // For now, delegate to MLIR built-in types
  // Future: implement custom neural network types
  return Type();
}

void NeuralNetworkDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  // For now, delegate to MLIR built-in types
  // Future: implement custom neural network type printing
}

//===----------------------------------------------------------------------===//
// Neural Network operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "nnc/IR/NeuralNetworkOps.cpp.inc"