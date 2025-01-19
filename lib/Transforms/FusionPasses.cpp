//===- FusionPasses.cpp - Operator fusion passes implementation -*- C++ -*-===//
//
// Neural Network Graph Compiler - Fusion Optimization Passes
//
//===----------------------------------------------------------------------===//

#include "nnc/Transforms/OperatorFusion.h"
#include "nnc/IR/NeuralNetworkDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace nnc {
namespace transforms {

using namespace mlir;
using namespace nnc::nn;

//===----------------------------------------------------------------------===//
// Fusion pattern implementations
//===----------------------------------------------------------------------===//

struct ConvReluFusionPattern : public OpRewritePattern<NN_ReluOp> {
  using OpRewritePattern<NN_ReluOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(NN_ReluOp reluOp, PatternRewriter &rewriter) const override {
    // Check if ReLU input comes from a Conv operation
    auto convOp = reluOp.getInput().getDefiningOp<NN_ConvOp>();
    if (!convOp) {
      return failure();
    }
    
    // Check if Conv has only one use (the ReLU)
    if (!convOp.getOutput().hasOneUse()) {
      return failure();
    }
    
    // Create fused Conv+ReLU operation
    rewriter.replaceOpWithNewOp<NN_FusedConvReluOp>(
        reluOp, reluOp.getType(), convOp.getInput(), convOp.getFilter(),
        convOp.getStrides(), convOp.getPadding());
    
    // Remove the original Conv operation
    rewriter.eraseOp(convOp);
    
    return success();
  }
};

struct MatMulAddFusionPattern : public OpRewritePattern<NN_AddOp> {
  using OpRewritePattern<NN_AddOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(NN_AddOp addOp, PatternRewriter &rewriter) const override {
    // Check if one input comes from MatMul
    auto matmulOp = addOp.getLhs().getDefiningOp<NN_MatMulOp>();
    if (!matmulOp) {
      matmulOp = addOp.getRhs().getDefiningOp<NN_MatMulOp>();
      if (!matmulOp) {
        return failure();
      }
    }
    
    // Check if MatMul has only one use
    if (!matmulOp.getOutput().hasOneUse()) {
      return failure();
    }
    
    // For now, we don't have a FusedMatMulAdd operation defined,
    // but this demonstrates the pattern
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Fusion pass implementation
//===----------------------------------------------------------------------===//

struct OperatorFusionPass : public PassWrapper<OperatorFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperatorFusionPass)
  
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();
    
    RewritePatternSet patterns(context);
    patterns.add<ConvReluFusionPattern, MatMulAddFusionPattern>(context);
    
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
  
  StringRef getArgument() const final { return "nn-operator-fusion"; }
  StringRef getDescription() const final {
    return "Fuse neural network operators for better performance";
  }
};

//===----------------------------------------------------------------------===//
// Memory layout optimization pass
//===----------------------------------------------------------------------===//

struct MemoryLayoutOptimizationPass : public PassWrapper<MemoryLayoutOptimizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryLayoutOptimizationPass)
  
  void runOnOperation() override {
    auto module = getOperation();
    
    // Walk through all operations and analyze memory layout opportunities
    module.walk([&](Operation *op) {
      if (auto convOp = dyn_cast<NN_ConvOp>(op)) {
        optimizeConvMemoryLayout(convOp);
      } else if (auto matmulOp = dyn_cast<NN_MatMulOp>(op)) {
        optimizeMatMulMemoryLayout(matmulOp);
      }
    });
  }
  
private:
  void optimizeConvMemoryLayout(NN_ConvOp convOp) {
    // Add attributes for optimized memory layout
    convOp->setAttr("memory_layout", StringAttr::get(convOp.getContext(), "NCHW_optimized"));
    convOp->setAttr("vectorization", BoolAttr::get(convOp.getContext(), true));
  }
  
  void optimizeMatMulMemoryLayout(NN_MatMulOp matmulOp) {
    // Add attributes for blocked memory layout
    matmulOp->setAttr("memory_layout", StringAttr::get(matmulOp.getContext(), "blocked"));
    matmulOp->setAttr("vectorization", BoolAttr::get(matmulOp.getContext(), true));
  }
  
  StringRef getArgument() const final { return "nn-memory-layout"; }
  StringRef getDescription() const final {
    return "Optimize memory layout for neural network operations";
  }
};

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

void registerNeuralNetworkPasses() {
  PassRegistration<OperatorFusionPass>();
  PassRegistration<MemoryLayoutOptimizationPass>();
}

} // namespace transforms
} // namespace nnc