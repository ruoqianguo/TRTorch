#include "core/conversion/conversionctx/ConversionCtx.h"
#include <iostream>
#include <sstream>
#include <utility>

namespace trtorch {
namespace core {
namespace conversion {

// clang-format off
std::ostream& operator<<(std::ostream& os, const BuilderSettings& s) {
    os << "Settings requested for TensorRT engine:"                                        \
       << "\n    Enabled Precisions: ";
       for (auto p = s.enabled_precisions.begin(); p != s.enabled_precisions.end(); ++p) {
        os << *p << ' ';
       }
    os << "\n    TF32 Floating Point Computation Enabled: " << !s.disable_tf32             \
       << "\n    Truncate Long and Double: " << s.truncate_long_and_double                 \
       << "\n    Make Refittable Engine: " << s.refit                                      \
       << "\n    Debuggable Engine: " << s.debug                                           \
       << "\n    Strict Types: " << s.strict_types                                         \
       << "\n    GPU ID: " << s.device.gpu_id                                              \
       << "\n    Allow GPU Fallback (if running on DLA): " << s.device.allow_gpu_fallback  \
       << "\n    Min Timing Iterations: " << s.num_min_timing_iters                        \
       << "\n    Avg Timing Iterations: " << s.num_avg_timing_iters                        \
       << "\n    Max Workspace Size: " << s.workspace_size;

    if (s.max_batch_size != 0) {
    os << "\n    Max Batch Size: " << s.max_batch_size;
    } else {
    os << "\n    Max Batch Size: Not set";
    }

    os << "\n    Device Type: " << s.device.device_type                                    \
       << "\n    GPU ID: " << s.device.gpu_id;
    if (s.device.device_type == nvinfer1::DeviceType::kDLA) {
    os << "\n    DLACore: " << s.device.dla_core;
    }
    os << "\n    Engine Capability: " << s.capability                                      \
       << "\n    Calibrator Created: " << (s.calibrator != nullptr);
    return os;
}
// clang-format on

ConversionCtx::ConversionCtx(BuilderSettings build_settings)
    : settings(build_settings),
      logger(
          "[TRTorch Conversion Context] - ",
          util::logging::get_logger().get_reportable_severity(),
          util::logging::get_logger().get_is_colored_output_on()) {
  // TODO: Support FP16 and FP32 from JIT information
  if (settings.device.gpu_id) {
    TRTORCH_CHECK(
        cudaSetDevice(settings.device.gpu_id) == cudaSuccess, "Unable to set gpu id: " << settings.device.gpu_id);
  }

  builder = make_trt(nvinfer1::createInferBuilder(logger));
  net = make_trt(
      builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

  LOG_INFO(settings);
  cfg = make_trt(builder->createBuilderConfig());

  for (auto p = settings.enabled_precisions.begin(); p != settings.enabled_precisions.end(); ++p) {
    switch (*p) {
      case nvinfer1::DataType::kHALF:
        TRTORCH_CHECK(builder->platformHasFastFp16(), "Requested inference in FP16 but platform does not support FP16");
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
        break;
      case nvinfer1::DataType::kINT8:
        TRTORCH_CHECK(builder->platformHasFastInt8(), "Requested inference in INT8 but platform does not support INT8");
        cfg->setFlag(nvinfer1::BuilderFlag::kINT8);
        if (!settings.calibrator) {
          LOG_INFO(
              "Int8 precision has been enabled but no calibrator provided. This assumes the network has Q/DQ nodes obtained from Quantization aware training. For more details, refer to https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks");
        } else {
          cfg->setInt8Calibrator(settings.calibrator);
        }
        break;
      case nvinfer1::DataType::kFLOAT:
        break;
      case nvinfer1::DataType::kINT32:
      case nvinfer1::DataType::kBOOL:
      default:
        TRTORCH_THROW_ERROR(
            "Requested kernel precision that is unsupported: " << *p << " options are float, half, int8");
    }
  }

  enabled_precisions = settings.enabled_precisions;

  if (settings.disable_tf32) {
    cfg->clearFlag(nvinfer1::BuilderFlag::kTF32);
  }
#if NV_TENSORRT_MAJOR > 7
  if (settings.sparse_weights) {
    cfg->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
  }
#endif
  if (settings.refit) {
    cfg->setFlag(nvinfer1::BuilderFlag::kREFIT);
  }

  if (settings.debug) {
    cfg->setFlag(nvinfer1::BuilderFlag::kDEBUG);
  }

  if (settings.strict_types) {
    cfg->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
  }

  if (settings.device.allow_gpu_fallback) {
    cfg->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  }

  if (settings.max_batch_size != 0) {
    builder->setMaxBatchSize(settings.max_batch_size);
  }

  cfg->setMinTimingIterations(settings.num_min_timing_iters);
  cfg->setAvgTimingIterations(settings.num_avg_timing_iters);
  cfg->setMaxWorkspaceSize(settings.workspace_size);
  cfg->setDefaultDeviceType(settings.device.device_type);
  cfg->setEngineCapability(settings.capability);

  if (settings.device.device_type == nvinfer1::DeviceType::kDLA) {
    auto nbDLACores = builder->getNbDLACores();
    TRTORCH_CHECK(
        static_cast<int>(settings.device.dla_core) < nbDLACores,
        "Configured DLA Core ID: " << settings.device.dla_core
                                   << " not available. Total number of available DLA Cores: " << nbDLACores);
    TRTORCH_CHECK(
        settings.enabled_precisions.find(nvinfer1::DataType::kFLOAT) == settings.enabled_precisions.end(),
        "DLA supports only fp16 or int8 precision");
    cfg->setDLACore(settings.device.dla_core);
  }
}

ConversionCtx::~ConversionCtx() {
  for (auto ptr : builder_resources) {
    free(ptr);
  }
}

nvinfer1::ITensor* ConversionCtx::AssociateValueAndTensor(const torch::jit::Value* value, nvinfer1::ITensor* tensor) {
  tensor->setName(value->debugName().c_str());
  this->value_tensor_map[value] = tensor;
  return tensor;
}

torch::jit::IValue* ConversionCtx::AssociateValueAndIValue(const torch::jit::Value* value, torch::jit::IValue ivalue) {
  this->evaluated_value_map[value] = std::move(ivalue);
  return &this->evaluated_value_map[value];
}

std::string ConversionCtx::SerializeEngine() {
#if NV_TENSORRT_MAJOR > 7
  auto serialized_network = builder->buildSerializedNetwork(*net, *cfg);
  if (!serialized_network) {
    TRTORCH_THROW_ERROR("Building serialized network failed in TensorRT");
  }
#else
  auto engine = builder->buildEngineWithConfig(*net, *cfg);
  if (!engine) {
    TRTORCH_THROW_ERROR("Building TensorRT engine failed");
  }
  auto serialized_network = engine->serialize();
  engine->destroy();
#endif
  auto engine_str = std::string((const char*)serialized_network->data(), serialized_network->size());
  return engine_str;
}

bool ConversionCtx::CheckLayerAddition(const torch::jit::Node* n) {
  for (auto out : n->outputs()) {
    auto iter_t = this->value_tensor_map.find(out);
    if (iter_t == this->value_tensor_map.end()) {
      auto iter_iv = this->evaluated_value_map.find(out);
      if (iter_iv == this->evaluated_value_map.end()) {
        LOG_WARNING(
            "Node "
            << util::node_info(n) << " output: " << out->debugName()
            << " does not have a coresponding value or tensor, may potentially indicate a defective evaluator or converter");
        return false;
      }
    }
  }
  return true;
}

} // namespace conversion
} // namespace core
} // namespace trtorch
