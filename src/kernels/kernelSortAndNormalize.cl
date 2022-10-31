#include <clStructs.h>
#include <constants.cl>

#include <sortPopulation.cl>
#include <Equals.cl>
#include <normalize.cl>

__kernel void kernelSortAndNormalize(constant parametersForGPU* parameters, global float* globalPopulationsBase,
                    local float* localScore, local ushort* localIndexes,
                    global int* popNewIndex, global float* bestScore) {

    uint runID=get_global_id(RUN_ID_2D);

    uint localSize=get_local_size(INDIVIDUAL_ID_2D);
    uint localID=get_local_id(INDIVIDUAL_ID_2D);

    global float* globalPopulations = &(globalPopulationsBase[popNewIndex[POPULATION_PLACE] * parameters->globalPopulationsSize]); // get population place

    int chromStoreLen = parameters->chromStoreLen;
    int popMaxSize = parameters->popMaxSize;
    int tempII; // temp individual index

    if(popNewIndex[POPULATION_INDEX_START] == 0) {
        // Initial step:

        // Copy Good: [0,popSize)
        tempII = localID;
        while(tempII < parameters->popSize) {

            global float* individual = getIndividual(popMaxSize, runID, tempII, chromStoreLen, globalPopulations);

            // Local Copy:
            localScore[tempII] = individual[chromStoreLen - CHROM_SUBTRACT_FOR_SCORE];
            localIndexes[tempII] = (ushort)tempII;

            tempII += localSize;
        }
        // Fill the rest of the local array [popSize, sortLength):
        tempII = parameters->popSize + localID;
        while(tempII < parameters->sortLength) {

            localScore[tempII] = BIG_SCORE_BIGGER;

            tempII += localSize;
        }
        // Sort
        bitonicMergeSort(localSize, localID, parameters->sortLength, localScore, localIndexes);
        // Copy to change order from best first to best last
        local float* localScoreGood = &(localScore[parameters->localScoreArrayLength]); // After current scores
        local ushort* localIndexesGood = &(localIndexes[popMaxSize]); // After initial indexes
        int popSize = parameters->popSize;
        int popSizeM1 = popSize - 1;
        tempII = localID;
        while(tempII < popSize) {

            int newII = popSizeM1 - tempII;
            localScoreGood[newII] = localScore[tempII];
            localIndexesGood[newII] = localIndexes[tempII];

            tempII += localSize;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Normalize
        normalizeScore(localSize, localID, popSize, popMaxSize,
                  localScoreGood, localScore, bestScore, runID);
        // Copy
        copyPopulation(localSize, localID, popSize, popMaxSize,
                       localScoreGood, localScore, localIndexesGood,
                       globalPopulations, &(globalPopulationsBase[(popNewIndex[POPULATION_PLACE] == 0) * parameters->globalPopulationsSize]),
                       runID, chromStoreLen);
    } else {
        // All other steps:

        // Copy Good: [0,popMaxSize)
        tempII = localID;
        while(tempII < popMaxSize) {

            global float* individual = getIndividual(popMaxSize, runID, tempII, chromStoreLen, globalPopulations);
            
            // Local Copy:
            localScore[tempII] = individual[chromStoreLen - CHROM_SUBTRACT_FOR_SCORE];
            localIndexes[tempII] = (ushort)tempII;

            tempII += localSize;
        }
        // Fill the rest of the local array [popMaxSize, sortLength):
        tempII = popMaxSize + localID;
        while(tempII < parameters->sortLength) {

            localScore[tempII] = BIG_SCORE_BIGGER;

            tempII += localSize;
        }
        // Sort
        bitonicMergeSort(localSize, localID, parameters->sortLength, localScore, localIndexes);
        // Equals check and remove
        local ushort* localIndexesEquals = &(localIndexes[popMaxSize]); // After sort indexes
        tempII = localID;
        while(tempII < popMaxSize - 1) {
            // Check i and i+1 and if equals set i+1 as 0, that is why "popMaxSize - 1", NOT popMaxSize.

            // After a few steps, individuals should be more or less near each other.
            global float* individual1 = getIndividual(popMaxSize, runID, localIndexes[tempII], chromStoreLen, globalPopulations);
            global float* individual2 = getIndividual(popMaxSize, runID, localIndexes[tempII + 1], chromStoreLen, globalPopulations);

            int eq = EqualsGenome(individual1, individual2, parameters);// 0=Equals, 1=Good
            localIndexesEquals[tempII + 1] = eq;
            if(eq == 0) {
                localScore[tempII + 1] = BIG_SCORE_BIGGER; // Set as equals
            }

            tempII += localSize;
        }
        if(localID == 0) {
            localIndexesEquals[0] = 0;// Set first index as first
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Parallel Prefix Sum
        int lid = localID;
        int k;
        // phase 1
        for (k = 1; k <= (popMaxSize >> 1); k *= 2) {
            lid=localID;
            while (lid * k * 2 + 2 * k - 1 < popMaxSize) {
                localIndexesEquals[lid * k * 2 + 2 * k - 1] += localIndexesEquals[lid * k * 2 + k - 1];
                lid+=localSize;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // phase 2
        k *= 2;
        for (; k >= 2; k >>= 1) {
            lid = localID;  
            while (lid * k + k - 1 + (k >> 1) < popMaxSize) {
                localIndexesEquals[lid * k + k - 1 + (k >> 1)] += localIndexesEquals[lid * k + k - 1];
                lid+=localSize;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // Copy good ones (remove bad ones) and also move best scored (lowest number) individual to the end of array (now they are at the start).
        local float* localScoreGood = &(localScore[parameters->localScoreArrayLength]); // After current scores
        local ushort* localIndexesGood = &(localIndexes[2 * popMaxSize]); // After equals indexes
        int popSize = parameters->popSize;
        int popSizeM1 = popSize - 1;
        tempII = localID;
        while(tempII < popMaxSize) {

            // Copy only good and only popSize number
            if(localScore[tempII] != BIG_SCORE_BIGGER && localIndexesEquals[tempII] < popSize) {
                
                int newII = popSizeM1 - localIndexesEquals[tempII];
                localScoreGood[newII] = localScore[tempII];
                localIndexesGood[newII] = localIndexes[tempII];
            }

            tempII += localSize;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Normalize
        normalizeScore(localSize, localID, popSize, popMaxSize,
                  localScoreGood, localScore, bestScore, runID);
        // Copy
        copyPopulation(localSize, localID, popSize, popMaxSize,
                       localScoreGood, localScore, localIndexesGood,
                       globalPopulations, &(globalPopulationsBase[(popNewIndex[POPULATION_PLACE] == 0) * parameters->globalPopulationsSize]),
                       runID, chromStoreLen);
    }
}
