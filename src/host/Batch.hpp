#pragma once

#include <string>
#include <vector>

struct Batch {
    std::string inputPath;
    std::string outputPath;
    uint32_t platformIndex;
    uint32_t deviceIndex;
    uint32_t localSize;
    uint32_t timeKernels;
    uint32_t trackScores;
    uint32_t outputOnlyBestN;
    std::vector<std::string> jobs;
};

void parseBatch(int argc, char* argv[], Batch& batch);
