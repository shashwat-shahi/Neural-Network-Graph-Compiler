//===- NeuralNetworkDialect.h - Neural Network dialect --*- C++ -*-===//
//
// Neural Network Graph Compiler
//
//===----------------------------------------------------------------------===//

#ifndef NNC_DIALECT_NEURALNETWORK_H
#define NNC_DIALECT_NEURALNETWORK_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "nnc/IR/NeuralNetworkOpsDialect.h.inc"

namespace nnc {
namespace nn {

class NeuralNetworkDialect : public mlir::Dialect {
public:
  explicit NeuralNetworkDialect(mlir::MLIRContext *context);

  static constexpr llvm::StringLiteral getDialectNamespace() {
    return llvm::StringLiteral("nn");
  }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;
};

} // namespace nn
} // namespace nnc

#define GET_OP_CLASSES
#include "nnc/IR/NeuralNetworkOps.h.inc"

#endif // NNC_DIALECT_NEURALNETWORK_H