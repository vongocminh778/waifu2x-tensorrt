#ifndef WAIFU2X_TENSORRT_TRT_ENGINE_H
#define WAIFU2X_TENSORRT_TRT_ENGINE_H

#include <filesystem>
#include <fstream>
#include <memory>
#include <plog/Log.h>
#include <plog/Severity.h>
#include <string>
#include <NvOnnxParser.h>
#include <NvInfer.h>

#include "config.h"
#include "logger.h"

namespace trt {
    class SuperResEngine {
    public:
        SuperResEngine(BuilderConfig config);
        virtual ~SuperResEngine();
        bool load(const std::string& modelPath, InferrerConfig config);
        bool build(const std::string& onnxModelPath);

    private:
        Logger gLogger;
        BuilderConfig config;

        std::vector<float> input;
        std::vector<float> output;
        std::vector<std::pair<void*, int>> buffers;

        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        bool serializeConfig(std::string& onnxModelPath) const;
        static bool deserializeConfig(const std::string& trtEnginePath, BuilderConfig &trtEngineConfig);
        static void getDeviceNames(std::vector<std::string>& deviceNames);
    };
}

#endif //WAIFU2X_TENSORRT_TRT_ENGINE_H