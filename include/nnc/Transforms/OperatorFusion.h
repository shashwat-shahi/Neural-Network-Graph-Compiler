//===- OperatorFusion.h - Template metaprogramming for fusion -*- C++ -*-===//
//
// Neural Network Graph Compiler - Operator Fusion Framework
//
//===----------------------------------------------------------------------===//

#ifndef NNC_TRANSFORMS_OPERATORFUSION_H
#define NNC_TRANSFORMS_OPERATORFUSION_H

#include <type_traits>
#include <tuple>
#include <memory>

namespace nnc {
namespace fusion {

//===----------------------------------------------------------------------===//
// Type traits for fusion compatibility
//===----------------------------------------------------------------------===//

template<typename T>
struct is_fusable : std::false_type {};

template<typename T>
constexpr bool is_fusable_v = is_fusable<T>::value;

// Fusion pattern detection using template metaprogramming
template<typename Op1, typename Op2>
struct can_fuse : std::false_type {};

template<typename Op1, typename Op2>
constexpr bool can_fuse_v = can_fuse<Op1, Op2>::value;

//===----------------------------------------------------------------------===//
// Operator fusion templates
//===----------------------------------------------------------------------===//

template<typename... Ops>
class FusionPattern {
public:
  static constexpr size_t num_ops = sizeof...(Ops);
  using OpTuple = std::tuple<Ops...>;
  
  template<size_t I>
  using OpAt = std::tuple_element_t<I, OpTuple>;
  
  // Check if all operations in pattern are fusable
  static constexpr bool is_valid = (is_fusable_v<Ops> && ...);
  
  // Compile-time fusion validation
  template<size_t I = 0>
  static constexpr bool validate_fusion() {
    if constexpr (I + 1 < num_ops) {
      return can_fuse_v<OpAt<I>, OpAt<I + 1>> && validate_fusion<I + 1>();
    }
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Specific fusion patterns
//===----------------------------------------------------------------------===//

// Forward declarations for neural network operations
struct ConvOp;
struct ReluOp;
struct MatMulOp;
struct AddOp;

// Mark operations as fusable
template<> struct is_fusable<ConvOp> : std::true_type {};
template<> struct is_fusable<ReluOp> : std::true_type {};
template<> struct is_fusable<MatMulOp> : std::true_type {};
template<> struct is_fusable<AddOp> : std::true_type {};

// Define fusion compatibility
template<> struct can_fuse<ConvOp, ReluOp> : std::true_type {};
template<> struct can_fuse<MatMulOp, AddOp> : std::true_type {};
template<> struct can_fuse<AddOp, ReluOp> : std::true_type {};

//===----------------------------------------------------------------------===//
// Fusion execution engine
//===----------------------------------------------------------------------===//

template<typename FusionPattern>
class FusionExecutor {
public:
  static_assert(FusionPattern::is_valid, "Invalid fusion pattern");
  static_assert(FusionPattern::validate_fusion(), "Incompatible operations in pattern");
  
  template<typename... Args>
  auto execute(Args&&... args) -> decltype(auto) {
    return execute_impl<0>(std::forward<Args>(args)...);
  }
  
private:
  template<size_t I, typename... Args>
  auto execute_impl(Args&&... args) -> decltype(auto) {
    using CurrentOp = typename FusionPattern::template OpAt<I>;
    
    if constexpr (I + 1 < FusionPattern::num_ops) {
      // Intermediate operation - pass result to next operation
      auto result = CurrentOp::execute(std::forward<Args>(args)...);
      return execute_impl<I + 1>(result);
    } else {
      // Final operation - return result
      return CurrentOp::execute(std::forward<Args>(args)...);
    }
  }
};

//===----------------------------------------------------------------------===//
// Compile-time fusion pattern detection
//===----------------------------------------------------------------------===//

template<typename... Ops>
constexpr auto make_fusion_pattern() {
  using Pattern = FusionPattern<Ops...>;
  static_assert(Pattern::is_valid, "Operations are not fusable");
  static_assert(Pattern::validate_fusion(), "Operations cannot be fused in this order");
  return Pattern{};
}

// Convenience aliases for common patterns
using ConvReluPattern = FusionPattern<ConvOp, ReluOp>;
using MatMulAddPattern = FusionPattern<MatMulOp, AddOp>;
using MatMulAddReluPattern = FusionPattern<MatMulOp, AddOp, ReluOp>;

//===----------------------------------------------------------------------===//
// Memory layout optimization traits
//===----------------------------------------------------------------------===//

template<typename T>
struct memory_layout_traits {
  static constexpr bool supports_vectorization = false;
  static constexpr size_t alignment_requirement = 1;
  static constexpr bool supports_inplace = false;
};

template<typename Op>
constexpr bool supports_vectorization_v = memory_layout_traits<Op>::supports_vectorization;

template<typename Op>
constexpr size_t alignment_requirement_v = memory_layout_traits<Op>::alignment_requirement;

template<typename Op>
constexpr bool supports_inplace_v = memory_layout_traits<Op>::supports_inplace;

} // namespace fusion
} // namespace nnc

#endif // NNC_TRANSFORMS_OPERATORFUSION_H