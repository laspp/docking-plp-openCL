#include <clStructs.h>
#include <constants.cl>

#include <tyche_i.cl>
#include <classification.cl>
#include <PLP.cl>

__kernel void kernelInit(global ulong* seed, global tyche_i_state* rngStates,
                    constant parametersForGPU* parameters,
                    global AtomGPU* ligandAtoms, global AtomGPU* receptorAtoms,
                    global float* globalPopulations,
                    global LigandAtomPairsForClash* ligandAtomPairsForClash,
                    global DihedralRefDataGPU* dihedralRefData,
                    global BondGPU* ligandBonds) {

    uint runID=get_global_id(RUN_ID_2D);
    uint individualID=get_global_id(INDIVIDUAL_ID_2D);

    uint runSize=get_global_size(RUN_ID_2D);
    uint individualSize=get_global_size(INDIVIDUAL_ID_2D);

    // 1. SEED Random Number Generator and SAVE state
    if(individualID < parameters->popMaxSize) {
        tyche_i_state state;
        int RNGindex = getIndexOnePerIndividual(parameters->popMaxSize, runID, individualID);
        tyche_i_seed(&state,seed[RNGindex]);

        rngStates[RNGindex].a = state.a;
        rngStates[RNGindex].b = state.b;
        rngStates[RNGindex].c = state.c;
        rngStates[RNGindex].d = state.d;

        rngStates[RNGindex].res = state.res;
    }

    // 2.a Classify Ligand atoms
    int CLindex = getThreadIndex(individualSize, runID, individualID);
    while(CLindex < parameters->ligandNumAtoms) {

        ligandAtoms[CLindex].classification = getPLPclass(ligandAtoms[CLindex].triposType);
        WHILE_INCREMENT_2D(CLindex);
    }

    // 2.b Classify Receptor atoms
    int CRindex = getThreadIndex(individualSize, runID, individualID);
    while(CRindex < parameters->receptorNumAtoms) {

        receptorAtoms[CRindex].classification = getPLPclass(receptorAtoms[CRindex].triposType);
        receptorAtoms[CRindex].good = CsiteReceptor(receptorAtoms[CRindex].x, receptorAtoms[CRindex].y, receptorAtoms[CRindex].z, parameters);
        WHILE_INCREMENT_2D(CRindex);
    }

    // 3. Set rClash for ligandAtomPairsForClash
    int pairIndex = getThreadIndex(individualSize, runID, individualID);
    while(pairIndex < parameters->numLigandAtomPairsForClash) {

        int atom1ID = ligandAtomPairsForClash[pairIndex].atom1ID;
        int atom2ID = ligandAtomPairsForClash[pairIndex].atom2ID;
        int atom1Index = atom1ID - 1;
        int atom2Index = atom2ID - 1;

        if(ligandAtoms[atom1Index].id != atom1ID || ligandAtoms[atom2Index].id != atom2ID) {
            printf("[kernelInit] [kernelInit] [Atom1? ID(%d) and index(%d) MISMATCH?]\n", ligandAtoms[atom1Index].id, atom1Index);
            printf("[kernelInit] [kernelInit] [Atom2? ID(%d) and index(%d) MISMATCH?]\n", ligandAtoms[atom2Index].id, atom2Index);
        }
        ligandAtomPairsForClash[pairIndex].rClash = getRclash(getCLASHclass(ligandAtoms[atom1Index].triposType), getCLASHclass(ligandAtoms[atom2Index].triposType), ligandAtomPairsForClash[pairIndex].numBondsBetween);
        WHILE_INCREMENT_2D(pairIndex);
    }

    // 4. Set KS for Torsional
    int torsIndex = getThreadIndex(individualSize, runID, individualID);
    while(torsIndex < parameters->numDihedralElements) {

        setTorsionalKS(&(dihedralRefData[torsIndex]), ligandAtoms, ligandBonds);
        WHILE_INCREMENT_2D(torsIndex);
    }

    // 5. Set Individual Score to big values so first sort works correctly
    if(individualID < parameters->popMaxSize) {

        global float* individual = getIndividual(parameters->popMaxSize, runID, individualID, parameters->chromStoreLen, globalPopulations);
        individual[parameters->chromStoreLen - CHROM_SUBTRACT_FOR_SCORE] = BIG_SCORE_BIGGER;
    }
}
