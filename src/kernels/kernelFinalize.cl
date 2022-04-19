#include <clStructs.h>
#include <constants.cl>

#include <SyncToModel.cl>

__kernel void kernelFinalize(constant parametersForGPU* parameters, global float* globalPopulations,
                    global AtomGPUsmall* ligandAtomsSmallGlobalAll,
                    global DihedralRefDataGPU* dihedralRefData) {

    uint globalID=get_global_id(ID_1D);

    // Re-sync only the best from each run
    if(globalID < parameters->nruns) {

        global AtomGPUsmall* ligandAtomsOwn = getAtomGPUsmall(0, 0, globalID, parameters->ligandNumAtoms, ligandAtomsSmallGlobalAll);
        global float* individual = getIndividual(parameters->popMaxSize, globalID, parameters->popSize - 1, parameters->chromStoreLen, globalPopulations);
        syncToModel(ligandAtomsOwn, individual, dihedralRefData, parameters);
    }
}
