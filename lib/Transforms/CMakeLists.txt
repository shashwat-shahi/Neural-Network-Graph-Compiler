add_mlir_library(NeuralNetworkTransforms
  FusionPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nnc/Transforms

  DEPENDS
  NeuralNetworkOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  NeuralNetworkDialect
)