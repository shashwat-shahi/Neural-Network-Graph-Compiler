add_llvm_executable(nnc-opt nnc-opt.cpp)

target_link_libraries(nnc-opt PRIVATE
  NeuralNetworkDialect
  NeuralNetworkAnalysis
  NeuralNetworkTransforms
  NeuralNetworkUtils
  
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms
  
  LLVMSupport
)