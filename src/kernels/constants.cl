#ifndef CONSTANTS_CL_H
#define CONSTANTS_CL_H

#include <clStructs.h>

enum CHROM_SUBTRACT_FOR_CHROM {
    CHROM_SUBTRACT_FOR_RW_FITNESS = 1,
    CHROM_SUBTRACT_FOR_SCORE = 2,
    CHROM_SUBTRACT_FOR_ORIENTATION_3 = 3,
    CHROM_SUBTRACT_FOR_ORIENTATION_2 = 4,
    CHROM_SUBTRACT_FOR_ORIENTATION_1 = 5,
    CHROM_SUBTRACT_FOR_CENTER_OF_MASS_3 = 6,
    CHROM_SUBTRACT_FOR_CENTER_OF_MASS_2 = 7,
    CHROM_SUBTRACT_FOR_CENTER_OF_MASS_1 = 8
};
typedef enum CHROM_SUBTRACT_FOR_CHROM CHROM_SUBTRACT_FOR_CHROM;

enum CHROM_SUBTRACT_FOR_VALUE {
    CHROM_SUBTRACT_FOR_CHROM_ACTUAL_LENGTH = 2,
    CHROM_SUBTRACT_FOR_ROTATABLE_BONDS_LENGTH = 8
};
typedef enum CHROM_SUBTRACT_FOR_VALUE CHROM_SUBTRACT_FOR_VALUE;

// Ids:

#define ID_1D 0

#define RUN_ID_2D 1
#define INDIVIDUAL_ID_2D 0

#define CLASS_ID_2D 1
#define XYZ_ID_2D 0

#define RUN_ID_3D 2
#define INDIVIDUAL_ID_3D 1
#define RECEPTOR_ID_3D 0
#define NEW_INDIVIDUAL_ID_3D 0

// Indexes:

inline int getIndexOnePerIndividual(int popMaxSize, int runID, int individualID) {
    return runID * popMaxSize + individualID;
}

inline int getThreadIndex(int individualSize, int runID, int individualID) {
    return runID * individualSize + individualID;
}

#define WHILE_INCREMENT_2D(INDEX) INDEX+=runSize*individualSize

// Get From ONE from Glabal Array

inline global float* getIndividual(int popMaxSize, int runID, int individualID, int chromStoreLen, global float* globalPopulations) {
    return &(globalPopulations[runID * popMaxSize * chromStoreLen + chromStoreLen * individualID]);
}

inline global AtomGPUsmall* getAtomGPUsmall(int popMaxSize, int runID, int individualID, int ligandNumAtoms, global AtomGPUsmall* ligandAtomsSmallGlobalAll) {
    return &(ligandAtomsSmallGlobalAll[runID * popMaxSize * ligandNumAtoms + ligandNumAtoms * individualID]);
}

// Scoring:
__constant float BIG_SCORE = 99999.9f;
__constant float BIG_SCORE_BIGGER = 999999.9f;

#endif
