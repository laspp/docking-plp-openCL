#ifndef NORMALIZE_CL_H
#define NORMALIZE_CL_H

#include <clStructs.h>
#include <constants.cl>

void normalizeScore(const uint localSize, const uint localID, const int size, const int popMaxSize,
               local float* scores, local float* normalized, global float* bestScore, const uint runID) {

    float sum = 0.0f;
    float sumSq = 0.0f;
    float tempScore;

    int tempII;
    
    tempII = localID;
    while(tempII < size) {

        tempScore = scores[tempII];
        sum += tempScore;
        sumSq += tempScore * tempScore;

        tempII += localSize;
    }
    // minimun size normalized = 2 * localSize    [ sum | sumSq ]
    normalized[localID] = sum;
    normalized[localID + localSize] = sumSq;
    // Reduce Local,  Adapted from: https://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.reduction.2pp.pdf
    for (int offset = 1; offset < localSize; offset *= 2) {
        int mask = 2 * offset - 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((localID & mask) == 0) {
            normalized[localID] += normalized[localID + offset];// sum
            normalized[localID + localSize] += normalized[localID + localSize + offset];// sumSq
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    sum = normalized[0];
    sumSq = normalized[localSize];
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
    // TODO: FIXME: this is sequencial code...
    if(localID == 0) {

        for(int i=0; i < size; i++) {
            
            tempScore = scores[i];
            tempScoreRW = fmax(0.0f, tempScore - offset);

            tempScoreRW += partialSum;

            normalized[i] = tempScoreRW;

            partialSum = tempScoreRW;
        }
        normalized[size] = partialSum; // actual array length is max(popMaxSize, 2 * localSize), size=popSize, so arr[size] is OK.
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //NormaliseRWFitness
    float total = normalized[size];
    tempII = localID;
    while(tempII < size) {

        normalized[tempII] /= total;

        // set best score
        if(tempII == size - 1) {
            bestScore[runID] = scores[tempII];
        }

        tempII += localSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void copyPopulation(const uint localSize, const uint localID, const int size, const int popMaxSize,
               local float* scores, local float* normalized, local ushort* indexes,
               global float* globalPopulationsSrc, global float* globalPopulationsDest,
               const uint runID, const int chromStoreLen) {

    int tempII = localID;
    while(tempII < size) {

        global float* individualSrc = getIndividual(popMaxSize, runID, indexes[tempII], chromStoreLen, globalPopulationsSrc);
        global float* individualDest = getIndividual(popMaxSize, runID, tempII, chromStoreLen, globalPopulationsDest);

        // Global Copy Chromosome
        for(int i=0; i < chromStoreLen - CHROM_SUBTRACT_FOR_CHROM_ACTUAL_LENGTH; i++) {
            individualDest[i] = individualSrc[i];
        }
        // Copy score and normScore
        individualDest[chromStoreLen - CHROM_SUBTRACT_FOR_SCORE] = scores[tempII];
        individualDest[chromStoreLen - CHROM_SUBTRACT_FOR_RW_FITNESS] = normalized[tempII];

        tempII += localSize;
    }
}

#endif
