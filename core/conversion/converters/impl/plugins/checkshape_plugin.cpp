#include "checkshape_plugin.h"
#include "core/util/macros.h"

#define CUDA_MEM_ALIGN 256
#define NDEBUG
using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

namespace {
const char* CHECKSHAPE_PLUGIN_VERSION{"0"};
const char* CHECKSHAPE_PLUGIN_NAME{"CheckShapePlugin"};
const int CHECKSHAPE_PLUGIN_NUM_INPUT = 2;
const int CHECKSHAPE_PLUGIN_NUM_OUTPUT = 1;
} // namespace

// Write values into buffer
template <typename T>
void writeToBuffer(char*& buffer, const T& val) {
  *reinterpret_cast<T*>(buffer) = val;
  buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T readFromBuffer(const char*& buffer) {
  T val = *reinterpret_cast<const T*>(buffer);
  buffer += sizeof(T);
  return val;
}

// Calc aligned workspace size
inline size_t calcAlignedWsS(size_t wss) {
  size_t res = wss;
  if (wss % CUDA_MEM_ALIGN) {
    res += CUDA_MEM_ALIGN - (wss % CUDA_MEM_ALIGN);
  }
  return res;
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calcWorkspaceSize(size_t* workspaces, int count) {
  size_t total = 0;
  for (int i = 0; i < count; ++i) {
    total += calcAlignedWsS(workspaces[i]);
  }
  return total;
}

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int8_t*)addr;
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previousWorkspaceSize;
  return alignPtr((int8_t*)addr, CUDA_MEM_ALIGN);
}

// CheckShapePlugin
CheckShapePlugin::CheckShapePlugin(const std::string name, int in_rank, int expand_rank)
    : inputRank(in_rank), expandRank(expand_rank), mLayerName(name) {
  LOG_DEBUG("CheckShapePlugin::CheckShapePlugin(): this=" << this << inputRank << ", " << expandRank);
}

// used for deserialize from CheckShapePluginCreator::deserializePlugin
CheckShapePlugin::CheckShapePlugin(const std::string name, const void* data, size_t length) : mLayerName(name) {
  LOG_DEBUG("CheckShapePlugin::CheckShapePlugin() - for deserialize: this=" << this);
  const char* d = static_cast<const char*>(data);
  const char* a = d;

  inputRank = readFromBuffer<int>(d);
  expandRank = readFromBuffer<int>(d);

  assert(d == (a + length));
}

CheckShapePlugin::CheckShapePlugin(const CheckShapePlugin& obj) {
  LOG_DEBUG("CheckShapePlugin::CheckShapePlugin(CheckShapePlugin &obj): this=" << this);
  inputRank = obj.inputRank;
  expandRank = obj.expandRank;
  mLayerName = obj.mLayerName;
  mPluginNamespace = obj.mPluginNamespace;
}

CheckShapePlugin::~CheckShapePlugin() {
  LOG_DEBUG("CheckShapePlugin::~CheckShapePlugin(): this=" << this);
}

// inherited from IPluginV2
const char* CheckShapePlugin::getPluginType() const {
  return CHECKSHAPE_PLUGIN_NAME;
}

const char* CheckShapePlugin::getPluginVersion() const {
  return CHECKSHAPE_PLUGIN_VERSION;
}

inline int CheckShapePlugin::getNbOutputs() const {
  return CHECKSHAPE_PLUGIN_NUM_OUTPUT;
}

inline int CheckShapePlugin::initialize() {
  LOG_DEBUG("CheckShapePlugin::initialize(): this=" << this);
  return 0;
}

inline void CheckShapePlugin::terminate() {
  LOG_DEBUG("CheckShapePlugin::terminate(): this=" << this);
}

inline size_t CheckShapePlugin::getSerializationSize() const {
  size_t total_size = 0;
  total_size += sizeof(int); // inputRank
  total_size += sizeof(int); // expandRank
  return total_size;
}

inline void CheckShapePlugin::serialize(void* buffer) const {
  char* d = static_cast<char*>(buffer);
  const char* a = d;
  writeToBuffer<int>(d, inputRank);
  writeToBuffer<int>(d, expandRank);

  assert(d == (a + getSerializationSize()));
}

inline void CheckShapePlugin::destroy() {
  LOG_DEBUG("CheckShapePlugin::destroy(): this=" << this);
  delete this;
}

inline void CheckShapePlugin::setPluginNamespace(const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

inline const char* CheckShapePlugin::getPluginNamespace() const {
  return mPluginNamespace.c_str();
}

// inherited from IPluginV2Ext
inline DataType CheckShapePlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const {
  TRTORCH_ASSERT(nbInputs == CHECKSHAPE_PLUGIN_NUM_INPUT, "nbInputs should equal to CHECKSHAPE_PLUGIN_NUM_INPUT");
  TRTORCH_ASSERT(index < CHECKSHAPE_PLUGIN_NUM_OUTPUT, "output index should less than CHECKSHAPE_PLUGIN_NUM_OUTPUT");
  TRTORCH_ASSERT(inputTypes[0] == DataType::kINT32, "the datatype of input[0] should be kINT32");
  TRTORCH_ASSERT(inputTypes[1] == DataType::kINT32, "the datatype of input[1] should be kINT32");
  return DataType::kFLOAT;
}

inline void CheckShapePlugin::attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) {
}

inline void CheckShapePlugin::detachFromContext() {
}

// others
inline IPluginV2DynamicExt* CheckShapePlugin::clone() const {
  LOG_DEBUG("CheckShapePlugin::clone(): this=" << this);
  return new CheckShapePlugin(*this);
}

inline DimsExprs CheckShapePlugin::getOutputDimensions(
    int32_t outputIndex,
    const DimsExprs* inputs,
    int32_t nbInputs,
    IExprBuilder& exprBuilder) {
  LOG_DEBUG("CheckShapePlugin::getOutputDimensions: this=" << this);
  TRTORCH_ASSERT(inputs[0].nbDims == 1, "input[0]'s rank should be 1");
  TRTORCH_ASSERT(inputs[1].nbDims == 1, "input[1]'s rank should be 1");

  DimsExprs outputDims;
  outputDims.nbDims = inputRank;
  for(int i=0;i<inputRank;i++){
    outputDims.d[i] = exprBuilder.constant(1);
  }
  return outputDims;
}

inline bool CheckShapePlugin::supportsFormatCombination(
    int32_t pos,
    const PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) {
  switch (pos) {
    case 0: // input0
      return inOut[0].type == DataType::kINT32 && inOut[0].format == PluginFormat::kLINEAR;
    case 1: // input1
      return inOut[1].type == DataType::kINT32 && inOut[1].format == PluginFormat::kLINEAR;
    case 2: // output0
      return (inOut[2].type == DataType::kFLOAT) && inOut[2].format == PluginFormat::kLINEAR;
    default:
      return false;
  }
}

inline void CheckShapePlugin::configurePlugin(
    const DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const DynamicPluginTensorDesc* out,
    int32_t nbOutputs) {
  TRTORCH_ASSERT(inputRank == in[0].desc.dims.d[0], "input[0]'s dim should be (inputRank, )");
  TRTORCH_ASSERT(expandRank == in[1].desc.dims.d[0], "input[1]'s dim should be (expandRank, )");
}

inline size_t CheckShapePlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs,
    int32_t nbInputs,
    const PluginTensorDesc* outputs,
    int32_t nbOutputs) const {
  return 0;
}

/* When the input is dynamic, we can't validate the expansion in building engine time. So we utilize this plugin to validate the expansion. 
 The inputs of this plugin are dimensions of input tensor and the expanded sizes. This plugin mainly validates the expansion and only 
 dimensions of input tensor are actually used. To avoid TRT prunes away the plugin, we make a tensor as the output of this plugin which 
 has the same rank as input tensor and values are zero. The output will be added to the input tensor.
 Eg: Inputs are dimensions of input tensor [3, 1] and the expanded sizes [1, 3, 4], output is [[0.0],[0.0]]
*/
inline int32_t CheckShapePlugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) {
  LOG_DEBUG("CheckShapePlugin::enqueue()---------------------------------------------------------: this=" << this);
  int status = 0;

  int32_t* inShape = (int32_t*)inputs[0];
  int32_t* expandShape = (int32_t*)inputs[1];
  float* output = (float*)outputs[0];

  int* h_inShape = (int*)malloc(sizeof(int) * inputRank);
  int* h_expandShape = (int*)malloc(sizeof(int) * expandRank);
  cudaMemcpy(h_inShape, inShape, sizeof(int) * inputRank, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_expandShape, expandShape, sizeof(int) * expandRank, cudaMemcpyDeviceToHost);

  float* h_output = (float*)malloc(sizeof(float) * 1);
  *h_output = 0.0;
  cudaMemcpy(output, h_output, sizeof(float) * 1, cudaMemcpyHostToDevice);

  for (int i = expandRank - 1; i >= 0; i--) {
    int index_in = i - (expandRank - inputRank);
    if (index_in >= 0) {
      // Dimensions are right alignment. The expanded size of the tensor must match the existing size. Eg: an input of [3, 1] can be expanded to [1, 3, 4] but not [3, 4, 1]. 
      if (h_expandShape[i] != -1) {
        if (h_inShape[index_in] != 1) { 
          if(h_expandShape[i] != h_inShape[index_in]){
            LOG_ERROR("The expanded size of the tensor (" << h_expandShape[i] << ") must match the existing size ("
                                                  << h_inShape[index_in] << ") at non-singleton dimension " << i
                                                  << ". Target sizes: [" << h_expandShape << "].  Tensor sizes: ["
                                                  << h_inShape << "]");
            TRTORCH_THROW_ERROR("The expanded size of the tensor (" << h_expandShape[i] << ") must match the existing size ("
                                                  << h_inShape[index_in] << ") at non-singleton dimension " << i
                                                  << ". Target sizes: [" << h_expandShape << "].  Tensor sizes: ["
                                                  << h_inShape << "]");
            status = 1;
          }
        }
      }
    } else {
      // For the new dimensions, the size cannot be set to -1. Eg: an input of [3, 1] can't be expanded to [-1, 3, 4]. Passing -1 as the size for a dimension means not changing the size of that dimension.
      if(h_expandShape[i] < 0){
        LOG_ERROR("The expanded size of the tensor (" << h_expandShape[i]
                                              << ") isn't allowed in a leading, non-existing dimension " << i);
        TRTORCH_THROW_ERROR("The expanded size of the tensor (" << h_expandShape[i]
                                              << ") isn't allowed in a leading, non-existing dimension " << i);
        status = 2;
      }
    }
  }
  free(h_inShape);
  free(h_expandShape);
  free(h_output);
  return status;
}

// CheckShapePluginCreator
CheckShapePluginCreator::CheckShapePluginCreator() {
  LOG_DEBUG("CheckShapePluginCreator::CheckShapePluginCreator(): this=" << this);
}

inline const char* CheckShapePluginCreator::getPluginName() const {
  LOG_DEBUG("CheckShapePluginCreator::getPluginName(): this=" << this);
  return CHECKSHAPE_PLUGIN_NAME;
}

inline const char* CheckShapePluginCreator::getPluginVersion() const {
  LOG_DEBUG("CheckShapePluginCreator::getPluginVersion(): this=" << this);
  return CHECKSHAPE_PLUGIN_VERSION;
}

inline const PluginFieldCollection* CheckShapePluginCreator::getFieldNames() {
  LOG_DEBUG("CheckShapePluginCreator::getFieldNames(): this=" << this);
  LOG_DEBUG(__FUNCTION__);
  return nullptr;
}

IPluginV2* CheckShapePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) {
  LOG_DEBUG("CheckShapePluginCreator::createPlugin(): this=" << this);
  const PluginField* fields = fc->fields;
  // parse fields from PluginFieldCollection
  assert(fc->nbFields == 2);
  assert(fields[0].type == PluginFieldType::kINT32);
  int inputRank = *(static_cast<const int*>(fields[0].data));
  assert(fields[1].type == PluginFieldType::kINT32);
  int expandRank = *(static_cast<const int*>(fields[1].data));
  return new CheckShapePlugin(name, inputRank, expandRank);
}

IPluginV2* CheckShapePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) {
  LOG_DEBUG("CheckShapePluginCreator::deserializePlugin(): this=" << this);
  return new CheckShapePlugin(name, serialData, serialLength);
}

inline void CheckShapePluginCreator::setPluginNamespace(const char* pluginNamespace) {
  LOG_DEBUG("CheckShapePluginCreator::setPluginNamespace(): this=" << this);
  mPluginNamespace = pluginNamespace;
}

inline const char* CheckShapePluginCreator::getPluginNamespace() const {
  LOG_DEBUG("CheckShapePluginCreator::getPluginNamespace(): this=" << this);
  return mPluginNamespace.c_str();
}

// register plugin
REGISTER_TENSORRT_PLUGIN(CheckShapePluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch