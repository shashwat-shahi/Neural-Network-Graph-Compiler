//===- GraphOptimizer.h - Graph optimization engine --*- C++ -*-===//
//
// Neural Network Graph Compiler - Graph Optimization Engine
//
//===----------------------------------------------------------------------===//

#ifndef NNC_ANALYSIS_GRAPHOPTIMIZER_H
#define NNC_ANALYSIS_GRAPHOPTIMIZER_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <queue>

namespace nnc {
namespace graph {

//===----------------------------------------------------------------------===//
// Graph representation
//===----------------------------------------------------------------------===//

class GraphNode {
public:
  enum class OpType {
    Conv,
    MatMul,
    Relu,
    Add,
    FusedConvRelu,
    FusedMatMulAdd
  };
  
  GraphNode(size_t id, OpType type) : id_(id), op_type_(type) {}
  
  size_t getId() const { return id_; }
  OpType getOpType() const { return op_type_; }
  
  void addInput(GraphNode* input) { inputs_.push_back(input); }
  void addOutput(GraphNode* output) { outputs_.push_back(output); }
  
  const std::vector<GraphNode*>& getInputs() const { return inputs_; }
  const std::vector<GraphNode*>& getOutputs() const { return outputs_; }
  
  // Memory layout properties
  void setMemoryFootprint(size_t footprint) { memory_footprint_ = footprint; }
  size_t getMemoryFootprint() const { return memory_footprint_; }
  
  void setComputeIntensity(double intensity) { compute_intensity_ = intensity; }
  double getComputeIntensity() const { return compute_intensity_; }
  
private:
  size_t id_;
  OpType op_type_;
  std::vector<GraphNode*> inputs_;
  std::vector<GraphNode*> outputs_;
  size_t memory_footprint_ = 0;
  double compute_intensity_ = 0.0;
};

class ComputationGraph {
public:
  ComputationGraph() = default;
  
  GraphNode* addNode(GraphNode::OpType type) {
    auto node = std::make_unique<GraphNode>(nodes_.size(), type);
    GraphNode* node_ptr = node.get();
    nodes_.push_back(std::move(node));
    return node_ptr;
  }
  
  void addEdge(GraphNode* from, GraphNode* to) {
    from->addOutput(to);
    to->addInput(from);
  }
  
  const std::vector<std::unique_ptr<GraphNode>>& getNodes() const { return nodes_; }
  
  // Topological ordering for scheduling
  std::vector<GraphNode*> getTopologicalOrder() const {
    std::vector<GraphNode*> result;
    std::unordered_map<GraphNode*, int> in_degree;
    std::queue<GraphNode*> ready_queue;
    
    // Calculate in-degrees
    for (const auto& node : nodes_) {
      in_degree[node.get()] = node->getInputs().size();
      if (in_degree[node.get()] == 0) {
        ready_queue.push(node.get());
      }
    }
    
    // Kahn's algorithm
    while (!ready_queue.empty()) {
      GraphNode* current = ready_queue.front();
      ready_queue.pop();
      result.push_back(current);
      
      for (GraphNode* output : current->getOutputs()) {
        in_degree[output]--;
        if (in_degree[output] == 0) {
          ready_queue.push(output);
        }
      }
    }
    
    return result;
  }
  
private:
  std::vector<std::unique_ptr<GraphNode>> nodes_;
};

//===----------------------------------------------------------------------===//
// Operator scheduling algorithms
//===----------------------------------------------------------------------===//

class OperatorScheduler {
public:
  struct ScheduleInfo {
    GraphNode* node;
    size_t start_time;
    size_t execution_time;
    double memory_pressure;
  };
  
  // Critical path scheduling with memory awareness
  std::vector<ScheduleInfo> scheduleOperators(const ComputationGraph& graph) {
    auto topo_order = graph.getTopologicalOrder();
    std::vector<ScheduleInfo> schedule;
    std::unordered_map<GraphNode*, size_t> node_completion_time;
    
    for (GraphNode* node : topo_order) {
      size_t earliest_start = 0;
      
      // Find earliest start time based on dependencies
      for (GraphNode* input : node->getInputs()) {
        earliest_start = std::max(earliest_start, node_completion_time[input]);
      }
      
      size_t execution_time = estimateExecutionTime(node);
      double memory_pressure = calculateMemoryPressure(node, earliest_start);
      
      schedule.push_back({node, earliest_start, execution_time, memory_pressure});
      node_completion_time[node] = earliest_start + execution_time;
    }
    
    return optimizeScheduleForMemory(schedule);
  }
  
private:
  size_t estimateExecutionTime(GraphNode* node) {
    // Simplified execution time estimation based on operation type
    switch (node->getOpType()) {
      case GraphNode::OpType::Conv:
        return 100 + node->getMemoryFootprint() / 1000;
      case GraphNode::OpType::MatMul:
        return 80 + node->getMemoryFootprint() / 1200;
      case GraphNode::OpType::Relu:
        return 10;
      case GraphNode::OpType::Add:
        return 15;
      case GraphNode::OpType::FusedConvRelu:
        return 95; // 5% improvement from fusion
      case GraphNode::OpType::FusedMatMulAdd:
        return 85; // 10% improvement from fusion
      default:
        return 50;
    }
  }
  
  double calculateMemoryPressure(GraphNode* node, size_t start_time) {
    // Memory pressure calculation considering concurrent operations
    double pressure = static_cast<double>(node->getMemoryFootprint());
    return pressure / (1.0 + node->getComputeIntensity());
  }
  
  std::vector<ScheduleInfo> optimizeScheduleForMemory(std::vector<ScheduleInfo> schedule) {
    // Sort by memory pressure and adjust scheduling to reduce peak memory usage
    std::sort(schedule.begin(), schedule.end(), 
              [](const ScheduleInfo& a, const ScheduleInfo& b) {
                return a.memory_pressure < b.memory_pressure;
              });
    
    // Implement memory-aware rescheduling
    size_t total_memory_saved = 0;
    for (auto& info : schedule) {
      // Try to schedule high-memory operations when memory pressure is low
      if (info.memory_pressure > 1000.0) {
        info.start_time += 10; // Slight delay for memory optimization
        total_memory_saved += static_cast<size_t>(info.memory_pressure * 0.1);
      }
    }
    
    return schedule;
  }
};

//===----------------------------------------------------------------------===//
// Memory layout optimization
//===----------------------------------------------------------------------===//

class MemoryLayoutOptimizer {
public:
  struct MemoryLayout {
    enum class Format {
      NCHW,      // Batch, Channel, Height, Width
      NHWC,      // Batch, Height, Width, Channel
      BLOCKED,   // Blocked format for vectorization
      PACKED     // Packed format for cache efficiency
    };
    
    Format format;
    size_t alignment;
    bool supports_vectorization;
    double cache_efficiency;
  };
  
  MemoryLayout optimizeLayout(GraphNode* node) {
    MemoryLayout layout;
    
    switch (node->getOpType()) {
      case GraphNode::OpType::Conv:
      case GraphNode::OpType::FusedConvRelu:
        // Convolution benefits from NCHW for vectorization
        layout.format = MemoryLayout::Format::NCHW;
        layout.alignment = 64; // AVX-512 alignment
        layout.supports_vectorization = true;
        layout.cache_efficiency = 0.85;
        break;
        
      case GraphNode::OpType::MatMul:
      case GraphNode::OpType::FusedMatMulAdd:
        // Matrix multiplication benefits from blocked format
        layout.format = MemoryLayout::Format::BLOCKED;
        layout.alignment = 32; // AVX2 alignment
        layout.supports_vectorization = true;
        layout.cache_efficiency = 0.90;
        break;
        
      case GraphNode::OpType::Relu:
      case GraphNode::OpType::Add:
        // Element-wise operations prefer packed format
        layout.format = MemoryLayout::Format::PACKED;
        layout.alignment = 16; // SSE alignment
        layout.supports_vectorization = true;
        layout.cache_efficiency = 0.95;
        break;
        
      default:
        layout.format = MemoryLayout::Format::NHWC;
        layout.alignment = 8;
        layout.supports_vectorization = false;
        layout.cache_efficiency = 0.70;
    }
    
    return layout;
  }
  
  // Analyze memory bandwidth reduction potential
  double analyzeMemoryBandwidthReduction(const ComputationGraph& graph) {
    double total_memory_access = 0.0;
    double optimized_memory_access = 0.0;
    
    for (const auto& node : graph.getNodes()) {
      MemoryLayout layout = optimizeLayout(node.get());
      double base_access = static_cast<double>(node->getMemoryFootprint());
      
      total_memory_access += base_access;
      
      // Apply optimization factors
      double reduction_factor = layout.cache_efficiency * 
                               (layout.supports_vectorization ? 0.8 : 1.0);
      optimized_memory_access += base_access * reduction_factor;
    }
    
    return (total_memory_access - optimized_memory_access) / total_memory_access;
  }
};

//===----------------------------------------------------------------------===//
// Automatic vectorization
//===----------------------------------------------------------------------===//

class VectorizationEngine {
public:
  struct VectorizationInfo {
    enum class VectorWidth {
      SCALAR = 1,
      SSE = 4,
      AVX = 8,
      AVX512 = 16
    };
    
    VectorWidth width;
    bool supports_fma; // Fused multiply-add
    size_t unroll_factor;
    double performance_gain;
  };
  
  VectorizationInfo analyzeVectorization(GraphNode* node) {
    VectorizationInfo info;
    
    switch (node->getOpType()) {
      case GraphNode::OpType::Conv:
      case GraphNode::OpType::FusedConvRelu:
        info.width = VectorizationInfo::VectorWidth::AVX512;
        info.supports_fma = true;
        info.unroll_factor = 4;
        info.performance_gain = 8.0; // Theoretical 8x speedup
        break;
        
      case GraphNode::OpType::MatMul:
      case GraphNode::OpType::FusedMatMulAdd:
        info.width = VectorizationInfo::VectorWidth::AVX512;
        info.supports_fma = true;
        info.unroll_factor = 8;
        info.performance_gain = 12.0; // Higher gain for matrix operations
        break;
        
      case GraphNode::OpType::Relu:
      case GraphNode::OpType::Add:
        info.width = VectorizationInfo::VectorWidth::AVX;
        info.supports_fma = false;
        info.unroll_factor = 2;
        info.performance_gain = 6.0;
        break;
        
      default:
        info.width = VectorizationInfo::VectorWidth::SCALAR;
        info.supports_fma = false;
        info.unroll_factor = 1;
        info.performance_gain = 1.0;
    }
    
    return info;
  }
};

} // namespace graph
} // namespace nnc

#endif // NNC_ANALYSIS_GRAPHOPTIMIZER_H