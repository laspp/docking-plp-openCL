#include <clStructs.h>
#include <constants.cl>

__kernel void kernelInit3(constant parametersForGPU* parameters,
                    global AtomGPU* receptorAtoms,
                    global int* receptorIndex,
                    global int* numGoodReceptors,// one int
                    local int* localSum) {

    uint globalID=get_global_id(ID_1D);

    uint localID=get_local_id(ID_1D);
    uint localSize=get_local_size(ID_1D);
    
    // Set start and end index
    int size = parameters->receptorNumAtoms;
    int perThread = size / localSize + 1;// +1 to prevent, when lsize1=256 and size=500, last thread does almost half, but other problems

    int startIndex = localID * perThread;
    int endIndex = startIndex + perThread;

    // check border index
    if(startIndex < size && endIndex > size) {
        endIndex = size;
    }

    // Calculate own
    int count = 0;
    int i;
    for(i=startIndex; i < endIndex; i++) {
        if(i < size) {
            if(receptorAtoms[i].good == 1) {
                count++;
            }
        }
    }
    localSum[localID] = count;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Set correct offset (sequencial, only local_size)
    if(localID == 0) {
        for(i=1; i < localSize; i++) {// START WITH 1 !!
            localSum[i] += localSum[i - 1];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // get start index
    int tempIndex;
    if(localID == 0) {
        tempIndex = 0;
    } else {
        tempIndex = localSum[localID - 1];
    }
    // set index array
    for(i=startIndex; i < endIndex; i++) {
        if(i < size) {
            if(receptorAtoms[i].good == 1) {
                receptorIndex[tempIndex] = i;

                tempIndex++;
            }
        }
    }

    // set numGoodReceptors
    if(globalID == 0) {
        *numGoodReceptors = localSum[localSize - 1];
    }
}
