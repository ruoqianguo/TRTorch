#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>

#include "NvInfer.h"
#include "torch/csrc/jit/ir/ir.h"

#include <cuda_runtime.h>
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {

struct Device {
  nvinfer1::DeviceType device_type;
  int64_t gpu_id;
  int64_t dla_core;
  bool allow_gpu_fallback;
  Device() : device_type(nvinfer1::DeviceType::kGPU), gpu_id(0), dla_core(0), allow_gpu_fallback(false) {}
};

struct BuilderSettings {
  std::set<nvinfer1::DataType> enabled_precisions = {};
  bool sparse_weights = false;
  bool disable_tf32 = false;
  bool refit = false;
  bool debug = false;
  bool strict_types = false;
  bool truncate_long_and_double = false;
  Device device;
  nvinfer1::EngineCapability capability = TRT_ENGINE_CAPABILITY_STANDARD;
  nvinfer1::IInt8Calibrator* calibrator = nullptr;
  uint64_t num_min_timing_iters = 2;
  uint64_t num_avg_timing_iters = 1;
  uint64_t workspace_size = 0;
  uint64_t max_batch_size = 0;

  BuilderSettings() = default;
  BuilderSettings(const BuilderSettings& other) = default;
  friend std::ostream& operator<<(std::ostream& os, const BuilderSettings& s);
};

struct ConversionCtx {
  ConversionCtx(BuilderSettings settings);
  std::string SerializeEngine();
  nvinfer1::ITensor* AssociateValueAndTensor(const torch::jit::Value* value, nvinfer1::ITensor* tensor);
  torch::jit::IValue* AssociateValueAndIValue(const torch::jit::Value* value, torch::jit::IValue tensor);
  bool CheckLayerAddition(const torch::jit::Node* n);

  ~ConversionCtx();

  uint64_t num_inputs = 0;
  uint64_t num_outputs = 0;
  bool input_is_dynamic = false;
  std::shared_ptr<nvinfer1::IBuilder> builder;
  std::shared_ptr<nvinfer1::INetworkDefinition> net;
  std::shared_ptr<nvinfer1::IBuilderConfig> cfg;
  std::set<nvinfer1::DataType> enabled_precisions;
  BuilderSettings settings;
  util::logging::TRTorchLogger logger;
  // Pointers to data that needs to remain alive until conversion is done
  // All data will be freed when the destructor is called
  // The weights class is the main consumer of this, each time a weight object
  // is constructed from a PyTorch Tensor it allocates the data here to store a
  // copy of the values
  std::vector<void*> builder_resources;

  std::unordered_map<const torch::jit::Value*, nvinfer1::ITensor*> value_tensor_map;
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> evaluated_value_map;
};

} // namespace conversion
} // namespace core
} // namespace trtorch
