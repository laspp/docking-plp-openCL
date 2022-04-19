#include <clStructs.h>
#include <constants.cl>

#include <sortPopulation.cl>
#include <Equals.cl>

__kernel void kernelSort(constant parametersForGPU* parameters, global float* globalPopulations,
                    global float* globalPopulationsCopy,
                    local float* localScore, local ushort* localIndexes,
                    global int* popNewIndex) {

    uint runID=get_global_id(RUN_ID_2D);

    uint localID=get_local_id(INDIVIDUAL_ID_2D);
    uint localSize=get_local_size(INDIVIDUAL_ID_2D);

    int PNI = popNewIndex[0];

    // Sort Population

    // Set start and end index
    int size = parameters->popMaxSize;
    int perThread = size / localSize + 1;// +1 to prevent, when localSize=256 and size=500, last thread does almost half, but other problems

    int startIndex = localID * perThread;
    int endIndex = startIndex + perThread;

    // check border index
    if(startIndex < size && endIndex > size) {
        endIndex = size;
    }
    // Copy Local and Global
    int popMaxSize = parameters->popMaxSize;
    int chromStoreLen = parameters->chromStoreLen;

    for(int i=startIndex; i < endIndex && i < size; i++) {
        
        global float* individual = getIndividual(popMaxSize, runID, i, chromStoreLen, globalPopulations);
        global float* individualCopy = getIndividual(popMaxSize, runID, i, chromStoreLen, globalPopulationsCopy);

        // Local Copy:
        localScore[i] = individual[chromStoreLen - CHROM_SUBTRACT_FOR_SCORE];
        localIndexes[i] = (ushort)i;

        // Initial Step ONLY:
        if(PNI == 0) {
            // Global Copy (Whole)
            for(int j=0; j < chromStoreLen; j++) {
                individualCopy[j] = individual[j];
            }
        }
    }

    // Sort Local with bubble sort
    if(endIndex <= size) {
        bubblePopulationSort(startIndex, endIndex, localScore, localIndexes);
    }

    // Merge Sort
    // Reduce Local,  Adapted from: https://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.reduction.2pp.pdf
    int currentLength = perThread;
    int step01 = 1;
    for (int offset = 1; offset < localSize; offset *= 2) {
        int mask = 2 * offset - 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((localID & mask) == 0) {
            mergePopulationSort(startIndex, currentLength, step01, popMaxSize, size, localScore, localIndexes);
        }
        currentLength *= 2;
        step01 = (step01 == 0) ? 1 : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    step01 = (step01 == 0) ? 1 : 0;// Get Back Destination step01;

    if(PNI == 0) {
        // Initial Step ONLY

        // Copy Back Sorted
        int nReplicates = parameters->nReplicates;
        for(int i=startIndex; i < endIndex && i < size && i >= nReplicates; i++) {

            int copyDestination = (int)localIndexes[step01 * popMaxSize + i];

            global float* individualCopy = getIndividual(popMaxSize, runID, copyDestination, chromStoreLen, globalPopulationsCopy);
            global float* individual = getIndividual(popMaxSize, runID, i - nReplicates, chromStoreLen, globalPopulations);
            
            // Global Copy (Whole)
            for(int j=0; j < chromStoreLen; j++) {
                individual[j] = individualCopy[j];
            }
        }
    } else {
        // All other steps

        int sizeM1 = size - 1; // best at lowest index
        for(int i=startIndex; i < endIndex && i < size; i++) {

            int copyDestination = (int)localIndexes[step01 * popMaxSize + i];

            global float* individual = getIndividual(popMaxSize, runID, copyDestination, chromStoreLen, globalPopulations);
            global float* individualCopy = getIndividual(popMaxSize, runID, sizeM1 - i, chromStoreLen, globalPopulationsCopy);
            
            // Global Copy (Whole)
            for(int j=0; j < chromStoreLen; j++) {
                individualCopy[j] = individual[j];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Check i and i+1 and if equals set i+1 as 0, that is why sizeM1, NOT size.
        for(int i=startIndex; i < endIndex && i < sizeM1; i++) {

            global float* individual1 = getIndividual(popMaxSize, runID, i, chromStoreLen, globalPopulationsCopy);
            global float* individual2 = getIndividual(popMaxSize, runID, i + 1, chromStoreLen, globalPopulationsCopy);

            localIndexes[i+1] = EqualsGenome(individual1, individual2, parameters);// 0=Equals, 1=Good
            localIndexes[popMaxSize+i+1] = localIndexes[i+1];
        }

        if(localID == 0) {
            //localIndexes: [ isGood | PrefixSum(scan)-FinalIndex ]
            localIndexes[0] = 1;// Set first as Good
            localIndexes[popMaxSize] = 0;// Set first index as first
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Parallel Prefix Sum
        int lid = localID;
        int k;
        // phase 1
        for (k = 1; k <= (size >> 1); k *= 2) {
            lid=localID;
            while (lid * k * 2 + 2 * k - 1 < size) {
                localIndexes[popMaxSize+(lid * k * 2 + 2 * k - 1)] += localIndexes[popMaxSize+(lid * k * 2 + k - 1)];
                lid+=localSize;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // phase 2
        k *= 2;
        for (; k >= 2; k >>= 1) {
            lid = localID;  
            while (lid * k + k - 1 + (k >> 1) < size) {
                localIndexes[popMaxSize+(lid * k + k - 1 + (k >> 1))] += localIndexes[popMaxSize+(lid * k + k - 1)];
                lid+=localSize;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Copy good ones back
        int popSize = parameters->popSize;
        int popSizeM1 = popSize - 1;

        for(int i=startIndex; i < endIndex && i < size; i++) {

            if(localIndexes[i] == 1 && localIndexes[i+popMaxSize] < popSize) {// Copy only good and only popSize number

                global float* individualCopy = getIndividual(popMaxSize, runID, i, chromStoreLen, globalPopulationsCopy);
                global float* individual = getIndividual(popMaxSize, runID, popSizeM1 - localIndexes[i+popMaxSize], chromStoreLen, globalPopulations);
                
                // Global Copy (Whole)
                for(int j=0; j < chromStoreLen; j++) {
                    individual[j] = individualCopy[j];
                }
            }
        }
    }
}
