#include <clStructs.h>
#include <constants.cl>

#include <tyche_i.cl>
#include <rouletteWheelSelect.cl>
#include <mutate.cl>
#include <crossover.cl>

__kernel void kernelCreateNew(global tyche_i_state* rngStates,
                    constant parametersForGPU* parameters,
                    global float* globalPopulations,
                    global int* equalsArray,
                    global int* popNewIndex) {

    uint runID=get_global_id(RUN_ID_2D);
    uint individualID=get_global_id(INDIVIDUAL_ID_2D);

    if(individualID < parameters->nReplicatesNumThreads) {
        // Get RngState:
        tyche_i_state state;
        int RNGindex = runID * parameters->popMaxSize + individualID;
        state.a = rngStates[RNGindex].a;
        state.b = rngStates[RNGindex].b;
        state.c = rngStates[RNGindex].c;
        state.d = rngStates[RNGindex].d;

        state.res = rngStates[RNGindex].res;

        int popSize = parameters->popSize;
        
        // Set New Start and End Index
        // (matters only the first time it is run, switch from initial to new pop)
        if(runID == 0 && individualID == 0) {
            popNewIndex[0] = popSize;
            popNewIndex[1] = popSize + parameters->nReplicates;
        }

        // Reset equalsArray
        equalsArray[runID * parameters->nReplicates + 2 * individualID] = 0;
        equalsArray[runID * parameters->nReplicates + 2 * individualID + 1] = 0;

        // Get Father and Mother
        global float* father;
        global float* mother;
        global float* population = getIndividual(parameters->popMaxSize, runID, 0, parameters->chromStoreLen, globalPopulations);
        RouletteWheelSelect(population, &father, &mother, parameters->popSize, parameters->chromStoreLen, &state);

        // Get Children
        int child1Index = (2 * individualID + popSize) * parameters->chromStoreLen;
        int child2Index = child1Index + parameters->chromStoreLen;
        global float* child1 = &(population[child1Index]);
        global float* child2 = &(population[child2Index]);
        CopyParent2ToChild2(father, mother, child1, child2, parameters->chromStoreLen);

        // Crossover
        if (tyche_i_float(state) < parameters->pcrossover) {
            Crossover(father, mother, child1, child2, parameters->chromStoreLen, &state);
        
            // Cauchy mutation following crossover
            if (parameters->xovermut == 1) {
                CauchyMutate(child1, parameters->chromStoreLen, 0.0f, parameters->step_size, &state, parameters);
                CauchyMutate(child2, parameters->chromStoreLen, 0.0f, parameters->step_size, &state, parameters);
            }
        } else { // Mutation
            // Cauchy mutation
            if (parameters->cmutate == 1) {
                CauchyMutate(child1, parameters->chromStoreLen, 0.0f, parameters->step_size, &state, parameters);
                CauchyMutate(child2, parameters->chromStoreLen, 0.0f, parameters->step_size, &state, parameters);
            } else { // Regular mutation
                mutate(child1, parameters->chromStoreLen, parameters->step_size, &state, parameters);
                mutate(child2, parameters->chromStoreLen, parameters->step_size, &state, parameters);
            }
        }
        // No need to check for odd, since we forced even when calculating nReplicates.

        // Finalize (save RngState):
        rngStates[RNGindex].a = state.a;
        rngStates[RNGindex].b = state.b;
        rngStates[RNGindex].c = state.c;
        rngStates[RNGindex].d = state.d;

        rngStates[RNGindex].res = state.res;
    }
}
