add_mlir_dialect_library(NeuralNetworkDialect
  NeuralNetworkDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nnc/IR

  DEPENDS
  NeuralNetworkOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)