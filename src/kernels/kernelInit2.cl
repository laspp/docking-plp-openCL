#include <clStructs.h>
#include <constants.cl>

__kernel void kernelInit2(constant parametersForGPU* parameters,
                    global AtomGPU* ligandAtoms,
                    global AtomGPUsmall* ligandAtomsSmallGlobalAll) {

    uint runID=get_global_id(RUN_ID_2D);
    uint individualID=get_global_id(INDIVIDUAL_ID_2D);

    if(individualID < parameters->popMaxSize) {

        // Copy Own Ligand Atom Instance
        global AtomGPUsmall* ligandAtomsOwn = getAtomGPUsmall(parameters->popMaxSize, runID, individualID, parameters->ligandNumAtoms, ligandAtomsSmallGlobalAll);

        for(int i = 0; i < parameters->ligandNumAtoms; i++) {
            ligandAtomsOwn[i].id = ligandAtoms[i].id;
            ligandAtomsOwn[i].atomicNo = ligandAtoms[i].atomicNo;
            ligandAtomsOwn[i].triposType = ligandAtoms[i].triposType;
            ligandAtomsOwn[i].classification = ligandAtoms[i].classification;
            ligandAtomsOwn[i].atomicMass = ligandAtoms[i].atomicMass;
            ligandAtomsOwn[i].x = ligandAtoms[i].x;
            ligandAtomsOwn[i].y = ligandAtoms[i].y;
            ligandAtomsOwn[i].z = ligandAtoms[i].z;
        }
    }
}
