#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <string>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


int main()
{
    // =========================================================
    // 1. Paths
    // =========================================================

    const std::string imagePath =
        "../results/test_wafer.png";

    const std::wstring modelPath =
        L"../models/wafer_fault_cnn.onnx";


    // =========================================================
    // 2. Class Labels
    // =========================================================

    const std::vector<std::string> classNames =
    {
        "Center",
        "Donut",
        "Edge-Loc",
        "Edge-Ring",
        "Loc",
        "Near-full",
        "Random",
        "Scratch"
    };


    // =========================================================
    // 3. Load Image
    // =========================================================

    cv::Mat image = cv::imread(
        imagePath,
        cv::IMREAD_GRAYSCALE
    );


    if (image.empty())
    {
        std::cerr
            << "[ERROR] Failed to load image: "
            << imagePath
            << std::endl;

        return 1;
    }


    std::cout
        << "[SUCCESS] Image loaded: "
        << imagePath
        << std::endl;



    // =========================================================
    // 4. Preprocessing
    //
    // Python pipeline:
    // uint8 image
    // -> float32
    // -> normalize 0-1
    // -> resize 24x24
    // =========================================================


    cv::Mat waferMap;


    image.convertTo(
        waferMap,
        CV_32F
    );


    double minValue;
    double maxValue;


    cv::minMaxLoc(
        waferMap,
        &minValue,
        &maxValue
    );


    if (maxValue > 0.0)
    {
        waferMap /= static_cast<float>(maxValue);
    }


    cv::Mat resized;


    cv::resize(
        waferMap,
        resized,
        cv::Size(24, 24),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );


    // Ensure continuous memory
    resized = resized.clone();



    // =========================================================
    // 5. Create Input Tensor
    //
    // Shape:
    // [batch, channel, height, width]
    //
    // [1, 1, 24, 24]
    // =========================================================


    std::vector<float> inputData(
        24 * 24
    );


    std::copy(
        resized.ptr<float>(),
        resized.ptr<float>() + 24 * 24,
        inputData.begin()
    );


    std::array<int64_t, 4> inputShape =
    {
        1,
        1,
        24,
        24
    };



    // =========================================================
    // 6. Load ONNX Model
    // =========================================================


    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "WaferInference"
    );


    Ort::SessionOptions sessionOptions;


    sessionOptions.SetIntraOpNumThreads(
        1
    );


    Ort::Session session(
        env,
        modelPath.c_str(),
        sessionOptions
    );


    std::cout
        << "[SUCCESS] ONNX model loaded!"
        << std::endl;



    // =========================================================
    // 7. Create ONNX Tensor
    // =========================================================


    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );


    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputData.data(),
            inputData.size(),
            inputShape.data(),
            inputShape.size()
        );



    // =========================================================
    // 8. Run Inference
    // =========================================================


    Ort::AllocatorWithDefaultOptions allocator;


    auto inputName =
        session.GetInputNameAllocated(
            0,
            allocator
        );


    auto outputName =
        session.GetOutputNameAllocated(
            0,
            allocator
        );


    const char* inputNames[] =
    {
        inputName.get()
    };


    const char* outputNames[] =
    {
        outputName.get()
    };


    auto outputTensors =
        session.Run(
            Ort::RunOptions{ nullptr },
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );



    // =========================================================
    // 9. Find Prediction
    // =========================================================


    float* outputData =
        outputTensors[0]
        .GetTensorMutableData<float>();


    size_t outputCount =
        outputTensors[0]
        .GetTensorTypeAndShapeInfo()
        .GetElementCount();


    size_t predictedClass = 0;


    for (size_t i = 1; i < outputCount; ++i)
    {
        if (outputData[i] > outputData[predictedClass])
        {
            predictedClass = i;
        }
    }



    // =========================================================
    // 10. Display Result
    // =========================================================


    std::cout
        << "\n========================================\n";

    std::cout
        << "       Wafer Fault Prediction\n";

    std::cout
        << "========================================\n";


    for (size_t i = 0; i < outputCount; ++i)
    {
        std::cout
            << "Class "
            << i
            << " ("
            << classNames[i]
            << "): "
            << outputData[i]
            << std::endl;
    }


    std::cout
        << "\nPredicted class: "
        << predictedClass
        << " ("
        << classNames[predictedClass]
        << ")"
        << std::endl;


    std::cout
        << "========================================\n";


    return 0;
}
