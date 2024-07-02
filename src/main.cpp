#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "tensorrt/img2img.h"
#include "tensorrt/img2img.h"
#include "utilities/path.h"

int main(int argc, char *argv[]) {
    auto console = spdlog::stdout_color_mt("console");
    console->set_level(spdlog::level::info);
    console->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    // region Argument Parser
    CLI::App app("waifu2x-tensorrt");
    app.fallthrough()
        ->require_subcommand(1)
        ->get_formatter()->column_width(64);
    app.set_help_flag("", "");
    app.set_help_all_flag("-h, --help", "Print this help message and exit");

    std::string model;
    const auto modelChoices = {
        "cunet/art",
        "swin_unet/art",
        "swin_unet/art_scan",
        "swin_unet/photo",
        "upconv_7/photo"
    };
    app.add_option("--model", model)
        ->description("Set the model to use")
        ->check(CLI::IsMember(modelChoices))
        ->required();

    int scale;
    const auto scaleChoices = {
        1, 2, 4
    };
    app.add_option("--scale", scale)
        ->description("Set the scale factor")
        ->check(CLI::IsMember(scaleChoices))
        ->required();

    int noise;
    const auto noiseChoices = {
        -1, 0, 1, 2, 3
    };
    app.add_option("--noise", noise)
        ->description("Set the noise level")
        ->check(CLI::IsMember(noiseChoices))
        ->required();

    int batchSize;
    app.add_option("--batchSize", batchSize)
        ->description("Set the batch size")
        ->check(CLI::PositiveNumber)
        ->required();

    int tileSize;
    const auto tileSizeChoices = {
        64, 256, 400, 640
    };
    app.add_option("--tileSize", tileSize)
        ->description("Set the tile size")
        ->check(CLI::IsMember(tileSizeChoices))
        ->required();

    int deviceId = 0;
    app.add_option("--device", deviceId)
        ->description("Set the GPU device ID")
        ->default_val(deviceId)
        ->check(CLI::NonNegativeNumber);

    trt::Precision precision = trt::Precision::FP16;
    const std::map<std::string, trt::Precision> precisionMap = {
        {"fp16", trt::Precision::FP16},
        {"tf32", trt::Precision::TF32}
    };
    app.add_option("--precision", precision)
        ->description("Set the precision")
        ->default_val(precision)
        ->transform(CLI::CheckedTransformer(precisionMap, CLI::ignore_case));

    auto render = app.add_subcommand("render", "Render image(s)/video(s)");

    // std::vector<std::filesystem::path> inputPaths;
    // render->add_option("-i, --input", inputPaths)
    //     ->description("Set the input paths")
    //     ->check(CLI::ExistingPath)
    //     ->required();

    bool recursive = false;
    render->add_flag("--recursive", recursive)
        ->description("Search for input files recursively");

    std::filesystem::path outputDirectory;
    render->add_option("-o, --output", outputDirectory)
        ->description("Set the output directory")
        ->check(CLI::ExistingDirectory);

    double blend = 1.0/16.0;
    const auto blendChoices = {
        1.0/8.0, 1.0/16.0, 1.0/32.0, 0.0
    };
    render->add_option("--blend", blend)
        ->description("Set the percentage of overlap between two tiles to blend")
        ->default_val(blend)
        ->check(CLI::IsMember(blendChoices));

    bool tta = false;
    render->add_flag("--tta", tta)
        ->description("Enable test-time augmentation")
        ->default_val(tta);

    std::string codec = "libx264";
    render->add_option("--codec", codec)
        ->description("Set the codec (video only)")
        ->default_val(codec);

    std::string pixelFormat = "yuv420p";
    render->add_option("--pix_fmt", pixelFormat)
        ->description("Set the pixel format (video only)")
        ->default_val(pixelFormat);

    int crf = 23;
    render->add_option("--crf", crf)
        ->description("Set the constant rate factor (video only)")
        ->default_val(crf)
        ->check(CLI::Range(0, 51));

    auto build = app.add_subcommand("build", "Build model");

    try {
        app.parse((argc), (argv));
        if (model == "cunet/art" && scale == 4)
            throw std::runtime_error("cunet/art does not support scale factor 4.");
        if (noise == -1 && scale == 1)
            throw std::runtime_error("Noise level -1 does not support scale factor 1.");
    }
    catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    catch (const std::exception& e) {
        std::cerr << e.what();
        exit(-1);
    };
    // endregion

    trt::Img2Img engine;

    const auto modelPath = "models/" + model + "/"
        + (noise == -1 ? "" : "noise" + std::to_string(noise) + "_")
        + (scale == 1 ? "" : "scale" + std::to_string(scale) + "x")
        + ".onnx";
    std::replace(model.begin(), model.end(), '/', '_');
    const auto suffix = "(" + model + ")"
        + (noise == -1 ? "" : "(noise" + std::to_string(noise) + ")")
        + (scale == 1 ? "" : "(scale" + std::to_string(scale) + ")")
        + (tta ? "(tta)" : "");

    if (render->parsed()) {
        trt::RenderConfig config {
            .deviceId = deviceId,
            .precision = precision,
            .batchSize = batchSize,
            .channels = 3,
            .height = tileSize,
            .width = tileSize,
            .scaling = scale,
            .overlap = cv::Point2d(blend, blend),
            .tta = tta
        };

        if (!engine.load(modelPath, config))
            return -1;

        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        if (!cap.isOpened()) {
            console->error("Unable to open camera");
            return -1;
        }

        cv::Mat frame;
        cv::Mat outputFrame;
        double fps = 0.0;
        double tick_frequency = cv::getTickFrequency();
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                console->error("Empty frame captured");
                break;
            }

            outputFrame.create(frame.rows * scale, frame.cols * scale, frame.type());
            int64 frame_tick = cv::getTickCount();
            if (!engine.render(frame, outputFrame))
                return -1;
            int64 end_tick = cv::getTickCount();
            double frame_time = (end_tick - frame_tick) / tick_frequency;
            fps = 1.0 / frame_time;
            
            cv::Mat scaledFrame;
            cv::resize(frame, scaledFrame, cv::Size(frame.cols * 2, frame.rows * 2));

            // Display FPS on frame
            putText(outputFrame, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Original Scaled", scaledFrame);

            cv::imshow("Output", outputFrame);
            if (cv::waitKey(1) == 27) // Exit on ESC key
                break;
        }
        cap.release();
    } else if (build->parsed()) {
        trt::BuildConfig config {
            .deviceId = deviceId,
            .precision = precision,
            .minBatchSize = batchSize,
            .optBatchSize = batchSize,
            .maxBatchSize = batchSize,
            .minChannels = 3,
            .optChannels = 3,
            .maxChannels = 3,
            .minWidth = tileSize,
            .optWidth = tileSize,
            .maxWidth = tileSize,
            .minHeight = tileSize,
            .optHeight = tileSize,
            .maxHeight = tileSize,
        };
        if (!engine.build(modelPath, config))
            return -1;
    }

    return 0;
}
