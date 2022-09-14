#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include <vector>
#include <string>
#include "Batch.hpp"
#include "dbgl.hpp"
#include "../kernels/clStructs.h"
#include "Data.hpp"

#define MAX_PLATFORMS 10
#define MAX_DEVICES 10
#define MAX_SOURCE_SIZE 51200

class WorkerCL {

    void getStringVectorForKernels();

    Batch& batch;

public:

    // NOTE: when changing kernels enum CHANGE KernelNames strings in cpp ALSO, they MUST MATCH!
    enum Kernel {
        kernelInit=0,
        kernelInit2,
        kernelInit3,
        kernelInitGrid,
        kernelSyncToModel,
        kernelPLP,
        kernelSort,
        kernelNormalize,
        kernelCreateNew,
        kernelFinalize,
        NUM_KERNELS
    };
    static const std::vector<std::string> KernelNames;

    cl_int error;
    std::vector<std::string> programStrings;

    cl_context context;
    cl_command_queue commandQueue;
    cl_device_id* deviceID;
    cl_program program;
    cl_kernel kernels[Kernel::NUM_KERNELS];

    // Buffers:
    cl_mem cl_seed;
    cl_mem cl_rngStates;
    cl_mem cl_parameters;
    cl_mem cl_globalPopulations;
    cl_mem cl_globalPopulationsCopy;
    cl_mem cl_ligandAtoms;
    cl_mem cl_receptorAtoms;
    cl_mem cl_ligandBonds;
    cl_mem cl_ligandAtomsSmallGlobalAll;
    cl_mem cl_ligandAtomsSmallResult;
    cl_mem cl_dihedralRefData;
    cl_mem cl_equalsArray;
    cl_mem cl_receptorIndex;
    cl_mem cl_numGoodReceptors;
    cl_mem cl_bestScore;
    cl_mem cl_popNewIndex;
    cl_mem cl_ligandAtomPairsForClash;

    cl_mem cl_grid;

    // Sizes:
    size_t g_kernelInit[2];
    size_t l_kernelInit[2];
    size_t g_kernelInit2[2];
    size_t l_kernelInit2[2];
    size_t g_kernelInit3[1];
    size_t l_kernelInit3[1];
    size_t g_kernelInitGrid[2];
    size_t l_kernelInitGrid[2];
    size_t g_kernelSyncToModel[2];
    size_t l_kernelSyncToModel[2];
    size_t g_kernelPLP[2];
    size_t l_kernelPLP[2];
    size_t g_kernelSort[2];
    size_t l_kernelSort[2];
    size_t g_kernelNormalize[2];
    size_t l_kernelNormalize[2];
    size_t g_kernelCreateNew[2];
    size_t l_kernelCreateNew[2];
    size_t g_kernelFinalize[1];
    size_t l_kernelFinalize[1];

    // Timers:
    double t_programRunTime = 0.0;
    double tot_programRunTime = 0.0;
    double t_workerCreation = 0.0;
    double tot_workerCreation = 0.0;
    double t_kernelCreation = 0.0;
    double tot_kernelCreation = 0.0;
    double t_tempRunTime = 0.0;
    double tot_tempRunTime = 0.0;
    double* runTimes;
    uint32_t run = 0;

    WorkerCL(Batch& batchRef);
    ~WorkerCL();

    void initMemory(Data& data);
    void kernelCreation();
    void kernelSetArgs(Data& data);
    void initialStep(Data& data);
    void runStep(Data& data);
    void finalize(Data& data);
    void releaseMemory();

    void saveProgramTimersToFile(std::string path);
};
