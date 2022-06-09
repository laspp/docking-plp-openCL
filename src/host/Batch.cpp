#include "Batch.hpp"

#include "dbgl.hpp"
#include "io.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp> // Run CMake to fetch it.

void parseBatch(int argc, char* argv[], Batch& batch) {

    if (argc < 2) {
        dbglWE(__FILE__, __FUNCTION__, __LINE__, "No batch file provided as argument.");
    }

    std::ifstream batchIfs;
    openFile(std::string(argv[1]), batchIfs);

    nlohmann::json batchJson = nlohmann::json::parse(batchIfs, nullptr, false);
    if (batchJson.is_discarded()) {
        dbglWE(__FILE__, __FUNCTION__, __LINE__, "Invalid JSON in batch file.");
    }

    batch = {
        batchJson["inputPath"].get<std::string>(),
        batchJson["outputPath"].get<std::string>(),
        batchJson["platformIndex"].get<uint32_t>(),
        batchJson["deviceIndex"].get<uint32_t>(),
        batchJson["localSize"].get<uint32_t>(),
        batchJson["timeKernels"].get<uint32_t>(),
        batchJson["trackScores"].get<uint32_t>(),
        batchJson["outputOnlyBestN"].get<uint32_t>(),
        batchJson["jobs"].get<std::vector<std::string>>()
    };
    
}