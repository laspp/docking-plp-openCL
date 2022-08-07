#include <clStructs.h>
#include <constants.cl>

#include <SyncToModel.cl>

__kernel void kernelFinalize(constant parametersForGPU* parameters, global float* globalPopulations,
                    global AtomGPUsmall* ligandAtomsSmallGlobalAll,
                    global DihedralRefDataGPU* dihedralRefData,
                    global AtomGPUsmall* ligandAtomsSmallResult,
                    global AtomGPU* ligandAtoms) {

    uint globalID=get_global_id(ID_1D);

    // LIMIT: !!! nRuns < popMaxSize !!!

    // Re-sync only the best from each run
    if(globalID < parameters->nruns) {
        // Sync:
        global AtomGPUsmall* ligandAtomsOwn = getAtomGPUsmallBase(0, 0, globalID, parameters->ligandNumAtoms, ligandAtomsSmallGlobalAll);
        global float* individual = getIndividual(parameters->popMaxSize, globalID, parameters->popSize - 1, parameters->chromStoreLen, globalPopulations);
        syncToModel(ligandAtomsOwn, ligandAtoms, individual, dihedralRefData, parameters);

        // Copy to old style Array for Host.
        global AtomGPUsmall* ligandAtomsResult = &(ligandAtomsSmallResult[parameters->ligandNumAtoms * globalID]);
        for(int i = 0; i < parameters->ligandNumAtoms; i++) {
            global AtomGPUsmall* tempAtom = getAtomGPUsmallFromBase(parameters->popMaxSize, i, ligandAtomsOwn);
            ligandAtomsResult[i].x = tempAtom->x;
            ligandAtomsResult[i].y = tempAtom->y;
            ligandAtomsResult[i].z = tempAtom->z;
        }
    }
}
