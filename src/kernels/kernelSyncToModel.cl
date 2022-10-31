#include <clStructs.h>
#include <constants.cl>

#include <SyncToModel.cl>

__kernel void kernelSyncToModel(constant parametersForGPU* parameters, global float* globalPopulationsBase,
                    global AtomGPUsmall* ligandAtomsSmallGlobalAll,
                    global DihedralRefDataGPU* dihedralRefData,
                    global int* popNewIndex,
                    global AtomGPU* ligandAtoms) {

    uint runID=get_global_id(RUN_ID_2D);
    uint individualID=get_global_id(INDIVIDUAL_ID_2D);

    global float* globalPopulations = &(globalPopulationsBase[popNewIndex[POPULATION_PLACE] * parameters->globalPopulationsSize]); // get population place

    // What part of population to Sync (existing (only initial) or new pop) (ALL THREADS SAME PATH).
    if(popNewIndex[POPULATION_INDEX_START] != 0) {
        individualID += parameters->popSize;
    }

    // Sync to Model Individuals
    if(individualID >= popNewIndex[POPULATION_INDEX_START] && individualID < popNewIndex[POPULATION_INDEX_END]) {

        global AtomGPUsmall* ligandAtomsOwn = getAtomGPUsmallBase(parameters->popMaxSize, runID, individualID, parameters->ligandNumAtoms, ligandAtomsSmallGlobalAll);
        global float* individual = getIndividual(parameters->popMaxSize, runID, individualID, parameters->chromStoreLen, globalPopulations);
        syncToModel(ligandAtomsOwn, ligandAtoms, individual, dihedralRefData, parameters); 
    }
}
