#include <clStructs.h>
#include <constants.cl>

#include <SyncToModel.cl>

__kernel void kernelFinalize(constant parametersForGPU* parameters, global float* globalPopulations,
                    global AtomGPUsmall* ligandAtomsSmallGlobalAll,
                    global DihedralRefDataGPU* dihedralRefData,
                    global AtomGPUsmall* ligandAtomsSmallResult) {

    uint globalID=get_global_id(ID_1D);

    // LIMIT: !!! nRuns < popMaxSize !!!

    // Re-sync only the best from each run
    if(globalID < parameters->nruns) {
        // Sync:
        global AtomGPUsmall* ligandAtomsOwn = getAtomGPUsmallBase(0, 0, globalID, parameters->ligandNumAtoms, ligandAtomsSmallGlobalAll);
        global float* individual = getIndividual(parameters->popMaxSize, globalID, parameters->popSize - 1, parameters->chromStoreLen, globalPopulations);
        syncToModel(ligandAtomsOwn, individual, dihedralRefData, parameters);

        // Copy to old style Array for Host.
        global AtomGPUsmall* ligandAtoms = &(ligandAtomsSmallResult[parameters->ligandNumAtoms * globalID]);
        for(int i = 0; i < parameters->ligandNumAtoms; i++) {
            global AtomGPUsmall* tempAtom = getAtomGPUsmallFromBase(parameters->popMaxSize, i, ligandAtomsOwn);
            ligandAtoms[i].id = tempAtom->id;
            ligandAtoms[i].atomicNo = tempAtom->atomicNo;
            ligandAtoms[i].triposType = tempAtom->triposType;
            ligandAtoms[i].classification = tempAtom->classification;
            ligandAtoms[i].atomicMass = tempAtom->atomicMass;
            ligandAtoms[i].x = tempAtom->x;
            ligandAtoms[i].y = tempAtom->y;
            ligandAtoms[i].z = tempAtom->z;
        }
    }
}
