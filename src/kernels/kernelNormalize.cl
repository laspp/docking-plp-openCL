#include <clStructs.h>
#include <constants.cl>

__kernel void kernelNormalize(constant parametersForGPU* parameters, global float* globalPopulations,
                    local float* localScore, global float* bestScore) {

    uint runID=get_global_id(RUN_ID_2D);

    uint localSize=get_local_size(INDIVIDUAL_ID_2D);
    uint localID=get_local_id(INDIVIDUAL_ID_2D);

    // Normalize

    // Set start and end index
    int size = parameters->popSize;
    int perThread = size / localSize + 1;// +1 to prevent, when localSize=256 and size=500, last thread does almost half, but other problems

    int startIndex = localID * perThread;
    int endIndex = startIndex + perThread;
    
    // Set Individual score:
    int popMaxSize = parameters->popMaxSize;
    int chromStoreLen = parameters->chromStoreLen;

    float sum = 0.0f;
    float sumSq = 0.0f;
    float tempScore;

    int tempII = localID;
    while(tempII < size) {

        global float* individual = getIndividual(popMaxSize, runID, tempII, chromStoreLen, globalPopulations);
        tempScore = individual[chromStoreLen - CHROM_SUBTRACT_FOR_SCORE];

        sum += tempScore;
        sumSq += tempScore * tempScore;

        tempII += localSize;
    }

    // length localScore = 2 * localSize    [ sum | sumSq ]
    localScore[localID] = sum;
    localScore[localID + localSize] = sumSq;
    
    // Reduce Local,  Adapted from: https://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.reduction.2pp.pdf
    for (int offset = 1; offset < localSize; offset *= 2) {
        int mask = 2 * offset - 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((localID & mask) == 0) {
            localScore[localID] += localScore[localID + offset];// sum
            localScore[localID + localSize] += localScore[localID + localSize + offset];// sumSq
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    sum = localScore[0];
    sumSq = localScore[localSize];

    barrier(CLK_LOCAL_MEM_FENCE);

    float m_scoreMean = sum / size;
    float m_scoreVariance = (sumSq / size) - (m_scoreMean * m_scoreMean);
    float sigma = sqrt(m_scoreVariance);
    // calculate scaled fitness values using sigma truncation
    // Goldberg page 124
    float m_c = 2.0f;// Sigma Truncation Multiplier
    float offset = m_scoreMean - m_c * sigma;
    float partialSum = 0.0f;
    
    // First set the unnormalised fitness values
    float tempScoreRW;
    // FIXME: this is sequencial code...
    if(localID == 0) {

        for(int i=0; i < size; i++) {
            
            global float* individual = getIndividual(popMaxSize, runID, i, chromStoreLen, globalPopulations);

            tempScore = individual[chromStoreLen - CHROM_SUBTRACT_FOR_SCORE];
            tempScoreRW = fmax(0.0f, tempScore - offset);

            tempScoreRW += partialSum;

            individual[chromStoreLen - CHROM_SUBTRACT_FOR_RW_FITNESS] = tempScoreRW;

            partialSum = tempScoreRW;
        }
        localScore[0] = partialSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //NormaliseRWFitness
    float total = localScore[0];

    for(int i=startIndex; i < endIndex && i < size; i++) {
        
        global float* individual = getIndividual(popMaxSize, runID, i, chromStoreLen, globalPopulations);
        individual[chromStoreLen - CHROM_SUBTRACT_FOR_RW_FITNESS] /= total;

        // set best score
        if(i == size - 1) {
            bestScore[runID] = individual[chromStoreLen - CHROM_SUBTRACT_FOR_SCORE];
        }
    }
}
