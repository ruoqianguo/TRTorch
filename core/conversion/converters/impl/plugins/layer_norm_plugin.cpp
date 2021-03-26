#include "layer_norm_plugin.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

/*
 * LayerNormPlugin class implementations
 */

LayerNormPlugin::LayerNormPlugin(std::vector<int64_t> normalized_shape, float eps)
    : normalized_shape_(normalized_shape),
      eps_(eps) {
}

LayerNormPlugin::LayerNormPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);

  {
    torch::IValue value;
    input_archive.read("normalized_shape", value);
    normalized_shape_ = value.toIntVector();
  }
  {
    torch::IValue value;
    input_archive.read("eps", value);
    eps_ = value.toDouble();
  }
}

int LayerNormPlugin::getNbOutputs() const {
  return 1;
}

const char* LayerNormPlugin::getPluginType() const {
  return "LayerNorm";
}

const char* LayerNormPlugin::getPluginVersion() const {
  return "1";
}

const char* LayerNormPlugin::getPluginNamespace() const {
  return "";
}

nvinfer1::IPluginV2DynamicExt* LayerNormPlugin::clone() const {
  return new LayerNormPlugin(normalized_shape_, eps_);
}

nvinfer1::DimsExprs LayerNormPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

nvinfer1::DataType LayerNormPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    const {
  return inputTypes[0];
}

int LayerNormPlugin::initialize() {
  return 0;
}

void LayerNormPlugin::serialize(void* buffer) const {
  std::string data = serializeToString();
  size_t size = getSerializationSize();

  data.copy((char*)buffer, size);
}

std::string LayerNormPlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;

  output_archive.write("normalized_shape", torch::IValue(normalized_shape_));
  output_archive.write("eps", torch::IValue(eps_));

  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t LayerNormPlugin::getSerializationSize() const {
  return serializeToString().size();
}

bool LayerNormPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) {
  TRTORCH_ASSERT(0 <= pos && pos <= 3, "There should be exactly 4 connections to the plugin - 3 input, 1 output");
  TRTORCH_ASSERT(nbInputs == 3, "Expected a single tensor as input to LayerNorm plugin");
  TRTORCH_ASSERT(nbOutputs == 1, "Expected a single tensor as output to LayerNorm plugin");

  switch (pos)
  {
  case 0:  // input0
    /* code */
    return (inOut[0].type==DataType::kFLOAT || inOut[0].type==DataType::kHALF) && inOut[0].format==PluginFormat::kLINEAR;
  case 1:  // input1 weight
    /* code */
    return inOut[1].type==inOut[0].type && inOut[1].format==PluginFormat::kLINEAR;
  case 2:  // input2 bias
    /* code */
    return inOut[2].type==inOut[0].type && inOut[2].format==PluginFormat::kLINEAR;
  case 3:  // outpu0
    /* code */
    return inOut[3].type==inOut[0].type && inOut[3].format==PluginFormat::kLINEAR;
  
  default:
    return false;
  }
}

void LayerNormPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) {
}

size_t LayerNormPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const {
  return 0;
}

int LayerNormPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) {
  auto tensor_type = util::toATenDType(inputDesc[0].type);
  at::Tensor input= at::from_blob((void*)inputs[0], util::toVec(inputDesc[0].dims), [](void*) {}, at::device(at::kCUDA).dtype(tensor_type));
  at::Tensor weight = at::from_blob((void*)inputs[1], util::toVec(inputDesc[1].dims), [](void*) {}, at::device(at::kCUDA).dtype(tensor_type));
  at::Tensor bias = at::from_blob((void*)inputs[2], util::toVec(inputDesc[2].dims), [](void*) {},  at::device(at::kCUDA).dtype(tensor_type));
  at::Tensor output = at::from_blob(outputs[0], util::toVec(outputDesc[0].dims), [](void*) {}, at::device(at::kCUDA).dtype(tensor_type));

  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);
  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  at::Tensor output_ = at::layer_norm(input, normalized_shape_, weight, bias, eps_, false);
  output.copy_(output_);
  
  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);

  return 0;
}

/*
 * LayerNormPluginCreator class implementations
 */
const char* LayerNormPluginCreator::getPluginNamespace() const {
  return "";
}

const char* LayerNormPluginCreator::getPluginName() const {
  return "LayerNorm";
}

const char* LayerNormPluginCreator::getPluginVersion() const {
  return "1";
}

nvinfer1::IPluginV2* LayerNormPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
  return nullptr;
}

LayerNormPlugin* LayerNormPluginCreator::createPlugin(
    const char* name,
    std::vector<int64_t> normalized_shape,
    float eps) {
  name_ = name;
  return new LayerNormPlugin(normalized_shape, eps);
}

nvinfer1::IPluginV2* LayerNormPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) {
  name_ = name;
  return new LayerNormPlugin((const char*)serialData, serialLength);
}

const nvinfer1::PluginFieldCollection* LayerNormPluginCreator::getFieldNames() {
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch