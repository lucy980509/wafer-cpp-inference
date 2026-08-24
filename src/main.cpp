#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <chrono>

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
    // uint8
    // -> float32
    // -> normalize
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
        cv::Size(24,24),
        0,
        0,
        cv::INTER_LINEAR
    );


    resized = resized.clone();



    // =========================================================
    // 5. Create Input Tensor
    // Shape: [1,1,24,24]
    // =========================================================


    std::vector<float> inputData(
        24 * 24
    );


    std::copy(
        resized.ptr<float>(),
        resized.ptr<float>() + 24 * 24,
        inputData.begin()
    );


    std::array<int64_t,4> inputShape =
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


    sessionOptions.SetIntraOpNumThreads(1);


    Ort::Session session(
        env,
        modelPath.c_str(),
        sessionOptions
    );


    std::cout
        << "[SUCCESS] ONNX model loaded!"
        << std::endl;



    // =========================================================
    // 7. Create Tensor
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
    // 8. Prepare Inference
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



    // =========================================================
    // 9. Warm-up
    // =========================================================


    const int warmupRuns = 10;


    for(int i = 0; i < warmupRuns; i++)
    {
        session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );
    }



    // =========================================================
    // 10. ONNX Runtime Benchmark
    // =========================================================


    const int benchmarkRuns = 100;


    double totalLatency = 0.0;


    std::vector<Ort::Value> outputTensors;


    for(int i = 0; i < benchmarkRuns; i++)
    {

        auto start =
            std::chrono::high_resolution_clock::now();


        outputTensors =
            session.Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1
            );


        auto end =
            std::chrono::high_resolution_clock::now();


        totalLatency +=
            std::chrono::duration<double,std::milli>(
                end-start
            ).count();
    }


    double averageLatency =
        totalLatency / benchmarkRuns;


    std::cout
        << "\n========================================\n";

    std::cout
        << "          ONNX Runtime Benchmark\n";

    std::cout
        << "========================================\n";


    std::cout
        << "Average latency: "
        << averageLatency
        << " ms\n";


    std::cout
        << "FPS: "
        << 1000.0 / averageLatency
        << "\n";


    std::cout
        << "========================================\n";
    // =========================================================
    // 11. End-to-End Pipeline Benchmark
    //
    // Image
    // -> OpenCV preprocessing
    // -> Tensor creation
    // -> ONNX Runtime
    // -> Prediction
    // =========================================================


    const int pipelineRuns = 100;

    double totalPipelineLatency = 0.0;


    // output size
    size_t outputCount =
        outputTensors[0]
        .GetTensorTypeAndShapeInfo()
        .GetElementCount();



    for(int i = 0; i < pipelineRuns; i++)
    {

        auto pipelineStart =
            std::chrono::high_resolution_clock::now();



        // -----------------------------
        // Load image
        // -----------------------------

        cv::Mat pipelineImage =
            cv::imread(
                imagePath,
                cv::IMREAD_GRAYSCALE
            );


        // -----------------------------
        // Preprocessing
        // -----------------------------

        cv::Mat pipelineFloat;


        pipelineImage.convertTo(
            pipelineFloat,
            CV_32F
        );


        double pipelineMin;
        double pipelineMax;


        cv::minMaxLoc(
            pipelineFloat,
            &pipelineMin,
            &pipelineMax
        );


        if(pipelineMax > 0.0)
        {
            pipelineFloat /=
                static_cast<float>(pipelineMax);
        }


        cv::Mat pipelineResize;


        cv::resize(
            pipelineFloat,
            pipelineResize,
            cv::Size(24,24),
            0,
            0,
            cv::INTER_LINEAR
        );


        pipelineResize =
            pipelineResize.clone();



        // -----------------------------
        // Tensor creation
        // -----------------------------


        std::vector<float> pipelineInput(
            24 * 24
        );


        std::copy(
            pipelineResize.ptr<float>(),
            pipelineResize.ptr<float>() + 24 * 24,
            pipelineInput.begin()
        );



        Ort::Value pipelineTensor =
            Ort::Value::CreateTensor<float>(
                memoryInfo,
                pipelineInput.data(),
                pipelineInput.size(),
                inputShape.data(),
                inputShape.size()
            );



        // -----------------------------
        // ONNX Runtime inference
        // -----------------------------


        auto pipelineOutput =
            session.Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &pipelineTensor,
                1,
                outputNames,
                1
            );



        // -----------------------------
        // Prediction
        // -----------------------------


        float* pipelineOutputData =
            pipelineOutput[0]
            .GetTensorMutableData<float>();


        size_t pipelinePrediction = 0;


        for(size_t j = 1; j < outputCount; j++)
        {
            if(
                pipelineOutputData[j] >
                pipelineOutputData[pipelinePrediction]
            )
            {
                pipelinePrediction = j;
            }
        }



        auto pipelineEnd =
            std::chrono::high_resolution_clock::now();



        totalPipelineLatency +=
            std::chrono::duration<double,std::milli>(
                pipelineEnd - pipelineStart
            ).count();

    }



    double averagePipelineLatency =
        totalPipelineLatency / pipelineRuns;


    std::cout
        << "\n========================================\n";

    std::cout
        << "      End-to-End Pipeline Benchmark\n";

    std::cout
        << "========================================\n";


    std::cout
        << "Average latency: "
        << averagePipelineLatency
        << " ms"
        << std::endl;


    std::cout
        << "FPS: "
        << 1000.0 / averagePipelineLatency
        << std::endl;


    std::cout
        << "========================================\n";



    // =========================================================
    // 12. Prediction Result
    // =========================================================


    float* outputData =
        outputTensors[0]
        .GetTensorMutableData<float>();


    size_t predictedClass = 0;


    for(size_t i = 1; i < outputCount; i++)
    {
        if(outputData[i] > outputData[predictedClass])
        {
            predictedClass = i;
        }
    }



    // =========================================================
    // 13. Display Result
    // =========================================================


    std::cout
        << "\n========================================\n";


    std::cout
        << "       Wafer Fault Prediction\n";


    std::cout
        << "========================================\n";


    for(size_t i = 0; i < outputCount; i++)
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
