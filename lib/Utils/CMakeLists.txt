# Placeholder for utils library
add_mlir_library(NeuralNetworkUtils
  
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nnc/Utils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)