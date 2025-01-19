add_mlir_library(NeuralNetworkAnalysis
  GraphOptimizer.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nnc/Analysis

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  LLVMSupport
)