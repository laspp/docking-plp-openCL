#include <clStructs.h>
#include <constants.cl>

#include <PLP.cl>
#include <grid.cl>

__kernel void kernelPLP(constant parametersForGPU* parameters,
                    global AtomGPUsmall* ligandAtomsSmallGlobalAll,
                    global float* globalPopulationsBase,
                    global int* popNewIndex,
                    global LigandAtomPairsForClash* ligandAtomPairsForClash,
                    global DihedralRefDataGPU* dihedralRefData,
                    global GridPointGPU* grid,
                    global AtomGPU* ligandAtoms) {

    uint runID=get_global_id(RUN_ID_2D);
    uint individualID=get_global_id(INDIVIDUAL_ID_2D);

    global float* globalPopulations = &(globalPopulationsBase[popNewIndex[POPULATION_PLACE] * parameters->globalPopulationsSize]); // get population place

    float score = 0.0f;

    // What part of population to Score (existing (only initial) or new pop) (ALL THREADS SAME PATH).
    if(popNewIndex[POPULATION_INDEX_START] != 0) {
        individualID += parameters->popSize;
    }

    // Score Individuals
    if(individualID >= popNewIndex[POPULATION_INDEX_START] && individualID < popNewIndex[POPULATION_INDEX_END]) {

        global AtomGPUsmall* ligandAtomsOwn = getAtomGPUsmallBase(parameters->popMaxSize, runID, individualID, parameters->ligandNumAtoms, ligandAtomsSmallGlobalAll);

        GridGPU ownGrid;

        ownGrid.gridStep[0] = parameters->grid.gridStep[0];
        ownGrid.gridStep[1] = parameters->grid.gridStep[1];
        ownGrid.gridStep[2] = parameters->grid.gridStep[2];

        ownGrid.gridMin[0] = parameters->grid.gridMin[0];
        ownGrid.gridMin[1] = parameters->grid.gridMin[1];
        ownGrid.gridMin[2] = parameters->grid.gridMin[2];

        ownGrid.SXYZ[0] = parameters->grid.SXYZ[0];
        ownGrid.SXYZ[1] = parameters->grid.SXYZ[1];
        ownGrid.SXYZ[2] = parameters->grid.SXYZ[2];

        float min[3];
        min[0] = parameters->dockingSiteInfo.minCavity.x;
        min[1] = parameters->dockingSiteInfo.minCavity.y;
        min[2] = parameters->dockingSiteInfo.minCavity.z;
        float max[3];
        max[0] = parameters->dockingSiteInfo.maxCavity.x;
        max[1] = parameters->dockingSiteInfo.maxCavity.y;
        max[2] = parameters->dockingSiteInfo.maxCavity.z;
        
        // f_plp and c_site term:
        for(int i=0; i < parameters->ligandNumAtoms; i++) {
            global AtomGPUsmall* tempAtom = getAtomGPUsmallFromBase(parameters->popMaxSize, i, ligandAtomsOwn);
            score += GetSmoothedValue(tempAtom, &(ligandAtoms[i]), (float*)min, (float*)max, parameters, &ownGrid, grid);
        }

        // f_clash term:
        int currentAtom = 0;
        for(int i=0; i < parameters->numUniqueAtom1PairsForClash; i++) {

            int numAtom1 = ligandAtomPairsForClash[currentAtom].numAtom1;

            global AtomGPUsmall* atom1 = getAtomGPUsmallFromBase(parameters->popMaxSize, ligandAtomPairsForClash[currentAtom].atom1ID - 1, ligandAtomsOwn);

            float atom1COORD[3];
            atom1COORD[0] = atom1->x;
            atom1COORD[1] = atom1->y;
            atom1COORD[2] = atom1->z;

            for(int a = currentAtom; a < currentAtom + numAtom1; a++) {
                global AtomGPUsmall* atom2 = getAtomGPUsmallFromBase(parameters->popMaxSize, ligandAtomPairsForClash[a].atom2ID - 1, ligandAtomsOwn);
                score += PLPclash((float*)atom1COORD, atom2, ligandAtomPairsForClash[a].rClash);
            }
            currentAtom += numAtom1;
        }

        global float* individual = getIndividual(parameters->popMaxSize, runID, individualID, parameters->chromStoreLen, globalPopulations);
        
        // f_tors term:
        for(int i=0; i < parameters->numDihedralElements; i++) {

            score += PLPtorsional(individual[i], dihedralRefData[i].k, dihedralRefData[i].s);
        }

        // Set score:
        individual[parameters->chromStoreLen - CHROM_SUBTRACT_FOR_SCORE] = score;
    }
}
