#include "WorkerCL.hpp"

#include <fstream>
#include "io.hpp"
#include <omp.h>

#define CHECK_CL_ERROR(clOperation) clOperation;if(error!=CL_SUCCESS){dbglWE(__FILE__,__FUNCTION__,__LINE__,"Failed to perform OpenCL operation, error code: "+std::to_string(error)+".");}
#define TIME_CL(cmd, t_t, tot_t) if(batch.timeKernels==1){t_t=omp_get_wtime();CHECK_CL_ERROR(cmd) CHECK_CL_ERROR(error=clFlush(commandQueue)) CHECK_CL_ERROR(error=clFinish(commandQueue)) tot_t+=omp_get_wtime()-t_t;}else{CHECK_CL_ERROR(cmd)}
#define TIMER_START(t_t) if(batch.timeKernels==1){t_t=omp_get_wtime();}
#define TIMER_END(t_t, tot_t) if(batch.timeKernels==1){tot_t+=omp_get_wtime()-t_t;}
#define ASIGN_SIZE_2D(arr, zero, one) arr[0]=zero;arr[1]=one
#define ASIGN_SIZE_3D(arr, zero, one, two) arr[0]=zero;arr[1]=one;arr[2]=two

const std::vector<std::string> WorkerCL::KernelNames = {
    "kernelInit",
    "kernelInit2",
    "kernelInit3",
    "kernelInitGrid",
    "kernelSyncToModel",
    "kernelPLP",
    "kernelSort",
    "kernelNormalize",
    "kernelCreateNew",
    "kernelFinalize"
};

void WorkerCL::getStringVectorForKernels() {

    uint32_t numberOfKernels = static_cast<std::underlying_type<Kernel>::type>(Kernel::NUM_KERNELS);
    programStrings.reserve(numberOfKernels);

    for (uint32_t i = 0; i < numberOfKernels; i++) {
        
        std::string kernelFilePath = "src/kernels/" + KernelNames.at(i) + ".cl";

        std::ifstream kernelIfs;
        openFile(kernelFilePath, kernelIfs);

        programStrings.emplace_back(std::string(std::istreambuf_iterator<char>(kernelIfs), std::istreambuf_iterator<char>()));
    }
}

WorkerCL::WorkerCL(Data& data, Batch& batch) : programStrings() {

    TIMER_START(data.t_workerCreation);

    cl_platform_id	platformID[MAX_PLATFORMS];
    deviceID = new cl_device_id[MAX_DEVICES];
    cl_uint			returnNumber;

    CHECK_CL_ERROR(error = clGetPlatformIDs(MAX_PLATFORMS, platformID, &returnNumber));
    if (returnNumber <= batch.platformIndex) {
        dbglWE(__FILE__,__FUNCTION__,__LINE__,"No platform at index: "+std::to_string(batch.platformIndex)+", number of platforms found: "+std::to_string(returnNumber)+".");
    }

    CHECK_CL_ERROR(error = clGetDeviceIDs(platformID[batch.platformIndex], CL_DEVICE_TYPE_ALL, MAX_DEVICES, deviceID, &returnNumber));
    if (returnNumber <= batch.deviceIndex) {
        dbglWE(__FILE__,__FUNCTION__,__LINE__,"No device at index: "+std::to_string(batch.deviceIndex)+", number of devices found: "+std::to_string(returnNumber)+".");
    }

    CHECK_CL_ERROR(context = clCreateContext(NULL, 1, &deviceID[batch.deviceIndex], NULL, NULL, &error));
    CHECK_CL_ERROR(commandQueue = clCreateCommandQueue(context, deviceID[batch.deviceIndex], 0, &error));

    getStringVectorForKernels();

    const char* kernelsChar2D[Kernel::NUM_KERNELS];
    uint32_t numberOfKernels = static_cast<std::underlying_type<Kernel>::type>(Kernel::NUM_KERNELS);
    for (uint32_t i = 0; i < numberOfKernels; i++) {
        kernelsChar2D[i] = programStrings[i].c_str();
    }

    CHECK_CL_ERROR(program = clCreateProgramWithSource(context, Kernel::NUM_KERNELS, kernelsChar2D, NULL, &error));
    // No CHECK_CL_ERROR => exit() => no build log
    error = clBuildProgram(program, 1, &deviceID[batch.deviceIndex], "-I src/kernels/", NULL, NULL);
    if(error != CL_SUCCESS) {
        dbgl("Failed to build OpenCL program, error code: " + std::to_string(error) + ".");
    }

    size_t buildLogLen;
    CHECK_CL_ERROR(error = clGetProgramBuildInfo(program, deviceID[batch.deviceIndex], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogLen));
    char* buildLog = new char[buildLogLen + 1];
    CHECK_CL_ERROR(error = clGetProgramBuildInfo(program, deviceID[batch.deviceIndex], CL_PROGRAM_BUILD_LOG, buildLogLen, buildLog, NULL));
    buildLog[buildLogLen] = '\0';
    if(buildLogLen > 0) {
        dbgl(buildLog);
    }

    delete[] buildLog;

    TIMER_END(data.t_workerCreation, data.tot_workerCreation);
}

WorkerCL::~WorkerCL() {

    uint32_t numberOfKernels = static_cast<std::underlying_type<Kernel>::type>(Kernel::NUM_KERNELS);
    for (uint32_t i = 0; i < numberOfKernels; i++) {
        CHECK_CL_ERROR(error = clReleaseKernel(kernels[i]));
    }
    CHECK_CL_ERROR(error = clReleaseProgram(program));

    CHECK_CL_ERROR(error = clReleaseMemObject(cl_seed));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_rngStates));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_parameters));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_globalPopulations));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_globalPopulationsCopy));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_ligandAtoms));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_receptorAtoms));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_ligandBonds));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_ligandAtomsSmallGlobalAll));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_ligandAtomsSmallResult));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_dihedralRefData));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_equalsArray));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_receptorIndex));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_numGoodReceptors));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_bestScore));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_popNewIndex));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_ligandAtomPairsForClash));
    CHECK_CL_ERROR(error = clReleaseMemObject(cl_grid));

    CHECK_CL_ERROR(error = clReleaseCommandQueue(commandQueue));
    CHECK_CL_ERROR(error = clReleaseContext(context));

    delete[] deviceID;
}

void WorkerCL::initMemory(Data& data, Batch& batch) {
    
    TIMER_START(data.t_dataToGPU);

    CHECK_CL_ERROR(cl_seed = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.seedSize, data.seed, &error));
    CHECK_CL_ERROR(cl_rngStates = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.rngStatesSize, data.rngStates, &error));
    CHECK_CL_ERROR(cl_parameters = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.parametersSize, &(data.parameters), &error));
    CHECK_CL_ERROR(cl_globalPopulations = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.globalPopulationsSize, data.globalPopulations, &error));
    CHECK_CL_ERROR(cl_globalPopulationsCopy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.globalPopulationsCopySize, data.globalPopulationsCopy, &error));
    CHECK_CL_ERROR(cl_ligandAtoms = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.ligandAtomsSize, data.ligandAtoms, &error));
    CHECK_CL_ERROR(cl_receptorAtoms = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.receptorAtomsSize, data.receptorAtoms, &error));
    CHECK_CL_ERROR(cl_ligandBonds = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.ligandBondsSize, data.ligandBonds, &error));
    CHECK_CL_ERROR(cl_ligandAtomsSmallGlobalAll = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.ligandAtomsSmallGlobalAllSize, data.ligandAtomsSmallGlobalAll, &error));
    CHECK_CL_ERROR(cl_ligandAtomsSmallResult = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.ligandAtomsSmallResultSize, data.ligandAtomsSmallGlobalAll, &error));
    CHECK_CL_ERROR(cl_dihedralRefData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.dihedralRefDataSize, data.dihedralRefData, &error));
    CHECK_CL_ERROR(cl_equalsArray = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.equalsArraySize, data.equalsArray, &error));
    CHECK_CL_ERROR(cl_receptorIndex = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.receptorIndexSize, data.receptorIndex, &error));
    CHECK_CL_ERROR(cl_numGoodReceptors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.numGoodReceptorsSize, &(data.numGoodReceptors), &error));
    CHECK_CL_ERROR(cl_bestScore = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.bestScoreSize, data.bestScore, &error));
    CHECK_CL_ERROR(cl_popNewIndex = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.popNewIndexSize, data.popNewIndex, &error));
    CHECK_CL_ERROR(cl_ligandAtomPairsForClash = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.ligandAtomPairsForClashSize, data.ligandAtomPairsForClash, &error));
    CHECK_CL_ERROR(cl_grid = clCreateBuffer(context, CL_MEM_READ_WRITE, data.gridSize, NULL, &error));

    TIMER_END(data.t_dataToGPU, data.tot_dataToGPU);
}

void WorkerCL::kernelCreation(Data& data, Batch& batch) {

    TIMER_START(data.t_kernelCreation);

    uint32_t numberOfKernels = static_cast<std::underlying_type<Kernel>::type>(Kernel::NUM_KERNELS);

    for (uint32_t i = 0; i < numberOfKernels; i++) {
        
        CHECK_CL_ERROR(kernels[i] = clCreateKernel(program, KernelNames[i].c_str(), &error));
    }

    TIMER_END(data.t_kernelCreation, data.tot_kernelCreation);
}

void WorkerCL::kernelSetArgs(Data& data, Batch& batch) {

    TIMER_START(data.t_kernelSetArgs);

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 0, sizeof(cl_mem), (void *)&cl_seed));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 1, sizeof(cl_mem), (void *)&cl_rngStates));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 2, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 3, sizeof(cl_mem), (void *)&cl_ligandAtoms));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 4, sizeof(cl_mem), (void *)&cl_receptorAtoms));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 5, sizeof(cl_mem), (void *)&cl_globalPopulations));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 6, sizeof(cl_mem), (void *)&cl_ligandAtomPairsForClash));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 7, sizeof(cl_mem), (void *)&cl_dihedralRefData));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit], 8, sizeof(cl_mem), (void *)&cl_ligandBonds));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit2], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit2], 1, sizeof(cl_mem), (void *)&cl_ligandAtoms));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit2], 2, sizeof(cl_mem), (void *)&cl_ligandAtomsSmallGlobalAll));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit3], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit3], 1, sizeof(cl_mem), (void *)&cl_receptorAtoms));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit3], 2, sizeof(cl_mem), (void *)&cl_receptorIndex));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit3], 3, sizeof(cl_mem), (void *)&cl_numGoodReceptors));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInit3], 4, data.LOCAL_SIZE * sizeof(CL_STRUCT_INT), NULL));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInitGrid], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInitGrid], 1, sizeof(cl_mem), (void *)&cl_grid));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInitGrid], 2, sizeof(cl_mem), (void *)&cl_receptorAtoms));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInitGrid], 3, sizeof(cl_mem), (void *)&cl_numGoodReceptors));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelInitGrid], 4, sizeof(cl_mem), (void *)&cl_receptorIndex));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSyncToModel], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSyncToModel], 1, sizeof(cl_mem), (void *)&cl_globalPopulations));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSyncToModel], 2, sizeof(cl_mem), (void *)&cl_ligandAtomsSmallGlobalAll));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSyncToModel], 3, sizeof(cl_mem), (void *)&cl_dihedralRefData));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSyncToModel], 4, sizeof(cl_mem), (void *)&cl_popNewIndex));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSyncToModel], 5, sizeof(cl_mem), (void *)&cl_ligandAtoms));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 1, sizeof(cl_mem), (void *)&cl_ligandAtomsSmallGlobalAll));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 2, sizeof(cl_mem), (void *)&cl_globalPopulations));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 3, sizeof(cl_mem), (void *)&cl_popNewIndex));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 4, sizeof(cl_mem), (void *)&cl_ligandAtomPairsForClash));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 5, sizeof(cl_mem), (void *)&cl_dihedralRefData));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 6, sizeof(cl_mem), (void *)&cl_grid));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelPLP], 7, sizeof(cl_mem), (void *)&cl_ligandAtoms));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSort], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSort], 1, sizeof(cl_mem), (void *)&cl_globalPopulations));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSort], 2, sizeof(cl_mem), (void *)&cl_globalPopulationsCopy));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSort], 3, 2 * data.parameters.popMaxSize * sizeof(cl_float), NULL));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSort], 4, 2 * data.parameters.popMaxSize * sizeof(cl_ushort), NULL));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelSort], 5, sizeof(cl_mem), (void *)&cl_popNewIndex));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelNormalize], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelNormalize], 1, sizeof(cl_mem), (void *)&cl_globalPopulations));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelNormalize], 2, 2 * data.LOCAL_SIZE * sizeof(cl_float), NULL));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelNormalize], 3, sizeof(cl_mem), (void *)&cl_bestScore));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelCreateNew], 0, sizeof(cl_mem), (void *)&cl_rngStates));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelCreateNew], 1, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelCreateNew], 2, sizeof(cl_mem), (void *)&cl_globalPopulations));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelCreateNew], 3, sizeof(cl_mem), (void *)&cl_equalsArray));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelCreateNew], 4, sizeof(cl_mem), (void *)&cl_popNewIndex));

    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelFinalize], 0, sizeof(cl_mem), (void *)&cl_parameters));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelFinalize], 1, sizeof(cl_mem), (void *)&cl_globalPopulations));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelFinalize], 2, sizeof(cl_mem), (void *)&cl_ligandAtomsSmallGlobalAll));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelFinalize], 3, sizeof(cl_mem), (void *)&cl_dihedralRefData));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelFinalize], 4, sizeof(cl_mem), (void *)&cl_ligandAtomsSmallResult));
    CHECK_CL_ERROR(error = clSetKernelArg(kernels[kernelFinalize], 5, sizeof(cl_mem), (void *)&cl_ligandAtoms));

    TIMER_END(data.t_kernelSetArgs, data.tot_kernelSetArgs);
}

void WorkerCL::initialStep(Data& data, Batch& batch) {

    ASIGN_SIZE_2D(g_kernelInit, (data.parameters.popMaxSize / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.nruns);
    ASIGN_SIZE_2D(l_kernelInit, data.LOCAL_SIZE, 1);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelInit], 2, NULL, g_kernelInit, l_kernelInit, 0, NULL, NULL), data.t_kernelInit, data.tot_kernelInit);
    
    ASIGN_SIZE_2D(g_kernelInit2, (data.parameters.popMaxSize / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.nruns);
	ASIGN_SIZE_2D(l_kernelInit2, data.LOCAL_SIZE, 1);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelInit2], 2, NULL, g_kernelInit2, l_kernelInit2, 0, NULL, NULL), data.t_kernelInit2, data.tot_kernelInit2);

    g_kernelInit3[0] = data.LOCAL_SIZE;
	l_kernelInit3[0] = data.LOCAL_SIZE;
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelInit3], 1, NULL, g_kernelInit3, l_kernelInit3, 0, NULL, NULL), data.t_kernelInit3, data.tot_kernelInit3);

    // Read numGoodReceptors (blocking)
    TIME_CL(error = clEnqueueReadBuffer(commandQueue, cl_numGoodReceptors, CL_TRUE, 0, data.numGoodReceptorsSize, &(data.numGoodReceptors), 0, NULL, NULL), data.t_dataToCPU, data.tot_dataToCPU);

    ASIGN_SIZE_2D(g_kernelInitGrid, (data.parameters.grid.N / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.grid.totalPLPClasses);
	ASIGN_SIZE_2D(l_kernelInitGrid, data.LOCAL_SIZE, 1);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelInitGrid], 2, NULL, g_kernelInitGrid, l_kernelInitGrid, 0, NULL, NULL), data.t_kernelInitGrid, data.tot_kernelInitGrid);

    ASIGN_SIZE_2D(g_kernelSyncToModel, (data.parameters.popSize / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.nruns);
	ASIGN_SIZE_2D(l_kernelSyncToModel, data.LOCAL_SIZE, 1);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelSyncToModel], 2, NULL, g_kernelSyncToModel, l_kernelSyncToModel, 0, NULL, NULL), data.t_kernelSyncToModel, data.tot_kernelSyncToModel);

    ASIGN_SIZE_2D(g_kernelPLP, (data.parameters.popSize / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.nruns);
	ASIGN_SIZE_2D(l_kernelPLP, data.LOCAL_SIZE, 1);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelPLP], 2, NULL, g_kernelPLP, l_kernelPLP, 0, NULL, NULL), data.t_kernelPLP, data.tot_kernelPLP);

    ASIGN_SIZE_2D(g_kernelSort, data.LOCAL_SIZE, data.parameters.nruns);
	ASIGN_SIZE_2D(l_kernelSort, data.LOCAL_SIZE, 1);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelSort], 2, NULL, g_kernelSort, l_kernelSort, 0, NULL, NULL), data.t_kernelSort, data.tot_kernelSort);

    ASIGN_SIZE_2D(g_kernelNormalize, data.LOCAL_SIZE, data.parameters.nruns);
	ASIGN_SIZE_2D(l_kernelNormalize, data.LOCAL_SIZE, 1);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelNormalize], 2, NULL, g_kernelNormalize, l_kernelNormalize, 0, NULL, NULL), data.t_kernelNormalize, data.tot_kernelNormalize);

    // Read initial best score (blocking)
    TIME_CL(error = clEnqueueReadBuffer(commandQueue, cl_bestScore, CL_TRUE, 0, data.bestScoreSize, data.bestScore, 0, NULL, NULL), data.t_dataToCPU, data.tot_dataToCPU);
    if(batch.trackScores == 1) {
        data.trackScore();
    }
    
    // Change size
    ASIGN_SIZE_2D(g_kernelSyncToModel, (data.parameters.nReplicates / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.nruns);
	ASIGN_SIZE_2D(g_kernelPLP, (data.parameters.nReplicates / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.nruns);

    // Set size
    ASIGN_SIZE_2D(g_kernelCreateNew, (data.parameters.nReplicatesNumThreads / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE, data.parameters.nruns);
	ASIGN_SIZE_2D(l_kernelCreateNew, data.LOCAL_SIZE, 1);

    g_kernelFinalize[0] = (data.parameters.nruns / data.LOCAL_SIZE)*data.LOCAL_SIZE + data.LOCAL_SIZE;
	l_kernelFinalize[0] = data.LOCAL_SIZE;
}

void WorkerCL::runStep(Data& data, Batch& batch) {

    // Create new individuals
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelCreateNew], 2, NULL, g_kernelCreateNew, l_kernelCreateNew, 0, NULL, NULL), data.t_kernelCreateNew, data.tot_kernelCreateNew);

    // Same as initialStep (without inits)
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelSyncToModel], 2, NULL, g_kernelSyncToModel, l_kernelSyncToModel, 0, NULL, NULL), data.t_kernelSyncToModel, data.tot_kernelSyncToModel);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelPLP], 2, NULL, g_kernelPLP, l_kernelPLP, 0, NULL, NULL), data.t_kernelPLP, data.tot_kernelPLP);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelSort], 2, NULL, g_kernelSort, l_kernelSort, 0, NULL, NULL), data.t_kernelSort, data.tot_kernelSort);
    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelNormalize], 2, NULL, g_kernelNormalize, l_kernelNormalize, 0, NULL, NULL), data.t_kernelNormalize, data.tot_kernelNormalize);
    
    // Read best score (blocking)
    TIME_CL(error = clEnqueueReadBuffer(commandQueue, cl_bestScore, CL_TRUE, 0, data.bestScoreSize, data.score, 0, NULL, NULL), data.t_dataToCPU, data.tot_dataToCPU);
    if(batch.trackScores == 1) {
        data.trackScore();
    }
    // FIXME: Check score/check reduced Bool var if finished when that kernel is implemented
}

void WorkerCL::finalize(Data& data, Batch& batch) {

    TIME_CL(error = clEnqueueNDRangeKernel(commandQueue, kernels[kernelFinalize], 1, NULL, g_kernelFinalize, l_kernelFinalize, 0, NULL, NULL), data.t_kernelFinalize, data.tot_kernelFinalize);

    // Read solution (blocking) (read only as much as needed)
    TIME_CL(error = clEnqueueReadBuffer(commandQueue, cl_ligandAtomsSmallResult, CL_TRUE, 0, data.ligandAtomsSmallResultSize, data.ligandAtomsSmallGlobalAll, 0, NULL, NULL), data.t_dataToCPU, data.tot_dataToCPU);
}
