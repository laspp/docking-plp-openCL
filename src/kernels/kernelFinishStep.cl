#include <clStructs.h>
#include <constants.cl>

// TODO: FIXME: find a better solution for this number change
__kernel void kernelFinishStep(global int* popNewIndex) {

    popNewIndex[POPULATION_PLACE] = (popNewIndex[POPULATION_PLACE] + 1) % 2;

}
