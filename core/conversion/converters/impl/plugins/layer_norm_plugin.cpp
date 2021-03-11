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
  return DataType::kFLOAT;
}

int LayerNormPlugin::initialize() {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  tensor_options_ = tensor_options_.device(c10::kCUDA);
#else
  tensor_options_ = tensor_options_.device(c10::kCPU);
#endif

  // c10::kFloat = FLOAT32
  tensor_options_ = tensor_options_.dtype(c10::kFloat);

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
  TRTORCH_ASSERT(nbInputs == 3, "Expected 3 tensors as input to LayerNorm plugin");
  TRTORCH_ASSERT(nbOutputs == 1, "Expected a single tensor as output to LayerNorm plugin");

  switch (pos)
  {
  case 0:  // input0
    /* code */
    return inOut[0].type==DataType::kFLOAT && inOut[0].format==PluginFormat::kLINEAR;
  case 1:  // input1 weight
    /* code */
    return inOut[1].type==DataType::kFLOAT && inOut[1].format==PluginFormat::kLINEAR;
  case 2:  // input2 bias
    /* code */
    return inOut[2].type==DataType::kFLOAT && inOut[2].format==PluginFormat::kLINEAR;
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
  dtype_ = DataType::kFLOAT;
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
  // #if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  //   at::Tensor input = at::from_blob((void*)inputs[0], util::toVec(inputDesc[0].dims), [](void*) {}, tensor_options_);
  //   at::Tensor weight = at::from_blob((void*)inputs[1], util::toVec(inputDesc[1].dims), [](void*) {}, tensor_options_);
  //   at::Tensor bias = at::from_blob((void*)inputs[2], util::toVec(inputDesc[2].dims), [](void*) {}, tensor_options_);
  //   at::Tensor output = at::from_blob(outputs[0], util::volume(outputDesc->dims), [](void*) {}, tensor_options_);

  //   at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  //   at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  //   cudaEvent_t event;
  //   cudaEventCreate(&event);
  //   cudaEventRecord(event, stream);

  //   cudaStreamWaitEvent(torch_stream.stream(), event, 0);

    

  //   cudaEvent_t torch_event;
  //   cudaEventCreate(&torch_event);
  //   cudaEventRecord(torch_event, torch_stream.stream());

  //   cudaStreamWaitEvent(stream, torch_event, 0);

  //   cudaEventDestroy(event);
  //   cudaEventDestroy(torch_event);

  //   return 0;
  // #else
    // TODO: When PyTorch updates to cuDNN 8 try moving back to CUDA based ATen
    // kernels HACK: WAR because there is a segfault if you try to create a CUDA
    // Tensor in the context of TensorRT execution.
    float* input_blob = (float*)malloc(util::volume(inputDesc[0].dims) * sizeof(float));
    float* weight_blob = (float*)malloc(util::volume(inputDesc[1].dims) * sizeof(float));
    float* bias_blob = (float*)malloc(util::volume(inputDesc[2].dims) * sizeof(float));
    cudaMemcpyAsync(
        input_blob,
        static_cast<const void*>(inputs[0]),
        util::volume(inputDesc[0].dims) * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    cudaMemcpyAsync(
        weight_blob,
        static_cast<const void*>(inputs[1]),
        util::volume(inputDesc[1].dims) * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    cudaMemcpyAsync(
        bias_blob,
        static_cast<const void*>(inputs[2]),
        util::volume(inputDesc[2].dims) * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    cudaStreamSynchronize(stream);

    at::Tensor input = at::from_blob((void*)input_blob, util::toVec(inputDesc[0].dims), tensor_options_);
    at::Tensor weight = at::from_blob((void*)weight_blob, util::toVec(inputDesc[1].dims), tensor_options_);
    at::Tensor bias = at::from_blob((void*)bias_blob, util::toVec(inputDesc[2].dims), tensor_options_);
    at::Tensor output;

    output = torch::layer_norm(input, normalized_shape_, weight, bias, eps_);

    cudaMemcpyAsync(
        outputs[0], output.data_ptr(), util::volume(outputDesc->dims) * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    free(input_blob);

    return 0;
  // #endif
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