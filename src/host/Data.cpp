#include "Data.hpp"

#include <fstream>
#include "io.hpp"
#include <cstdio>
#include "dbgl.hpp"
#include <memory>
#include <omp.h>
#include <algorithm>
#include <time.h>
#include <cmath>
#include <limits.h>

#define TIMER_START(t_t) if(batch.timeKernels==1){t_t=omp_get_wtime();}
#define TIMER_END(t_t, tot_t) if(batch.timeKernels==1){tot_t+=omp_get_wtime()-t_t;}

#define CAVITY_BORDER 0.0f
#define GOOD_RECEPTORS_BORDER 5.5f

#define TRIPOS_TYPE_H 19
#define TRIPOS_TYPE_H_P 20

void Data::initParameres(Header& header, LigandFlex& ligandFlex, GAParams& gaParams, CavityInfo& cavityInfo, uint32_t numDihedralElements, GridProps& gridProps) {

    parametersSize = sizeof(parametersForGPU);

    // Basic parameters
    parameters.nruns = (CL_STRUCT_INT)(header.nruns);
    parameters.popSize = (CL_STRUCT_INT)(header.popSize);
    // Ga parameters needed now: (rest bellow)
    parameters.new_fraction = (CL_STRUCT_FLOAT)(gaParams.new_fraction);
    parameters.nReplicates = (CL_STRUCT_INT)(parameters.new_fraction * (CL_STRUCT_FLOAT)parameters.popSize);
    if(parameters.nReplicates % 2 != 0) {// to make it divisible by 2
        parameters.nReplicates -= parameters.nReplicates % 2;
    }
    // end Ga needed
    parameters.popMaxSize = (CL_STRUCT_INT)(header.popSize + parameters.nReplicates);
    parameters.maxThreads = (CL_STRUCT_INT)(parameters.popMaxSize * parameters.nruns);
    parameters.chromStoreLen = (CL_STRUCT_INT)(header.chromStoreLen);
    parameters.maxDoubleLength = (CL_STRUCT_INT)(parameters.nruns * parameters.popMaxSize * parameters.chromStoreLen);
    // Number of Parameters
    parameters.ligandNumAtoms = (CL_STRUCT_INT)(header.ligandNumAtoms);
    parameters.ligandNumBonds = (CL_STRUCT_INT)(header.ligandNumBonds);
    parameters.receptorNumAtoms = (CL_STRUCT_INT)(header.receptorNumAtoms);
    parameters.receptorNumBonds = (CL_STRUCT_INT)(header.receptorNumBonds);
    // Ligand parameters
    parameters.dihedral_step = (CL_STRUCT_FLOAT)(ligandFlex.dihedral_step);
    parameters.rot_step = (CL_STRUCT_FLOAT)(ligandFlex.rot_step);
    parameters.trans_step = (CL_STRUCT_FLOAT)(ligandFlex.trans_step);
    parameters.max_dihedral = (CL_STRUCT_FLOAT)(ligandFlex.max_dihedral);
    parameters.max_rot = (CL_STRUCT_FLOAT)(ligandFlex.max_rot);
    parameters.max_trans = (CL_STRUCT_FLOAT)(ligandFlex.max_trans);
    parameters.transMode = (CL_STRUCT_INT)(ligandFlex.transMode);
    parameters.rotMode = (CL_STRUCT_INT)(ligandFlex.rotMode);
    parameters.dihedralMode = (CL_STRUCT_INT)(ligandFlex.dihedralMode);
    // GA Parameters TODO: Multiple different GA Runs
    parameters.history_freq = (CL_STRUCT_INT)(gaParams.history_freq);
    parameters.nconvergence = (CL_STRUCT_INT)(gaParams.nconvergence);
    parameters.ncycles = (CL_STRUCT_INT)(gaParams.ncycles);
    parameters.equality_threshold = (CL_STRUCT_FLOAT)(gaParams.equality_threshold);
    // ... Moved to Top
    parameters.nReplicatesNumThreads = (CL_STRUCT_INT)(parameters.nReplicates / 2);
    parameters.pcrossover = (CL_STRUCT_FLOAT)(gaParams.pcrossover);
    parameters.step_size = (CL_STRUCT_FLOAT)(gaParams.step_size);
    parameters.cmutate = (CL_STRUCT_INT)(gaParams.cmutate);
    parameters.xovermut = (CL_STRUCT_INT)(gaParams.xovermut);
    // Cavity Info
        // Principal Axes
    parameters.cavityInfo.principalAxes.center.x = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.center.x;
    parameters.cavityInfo.principalAxes.center.y = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.center.y;
    parameters.cavityInfo.principalAxes.center.z = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.center.z;

    parameters.cavityInfo.principalAxes.axis1.x = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis1.x;
    parameters.cavityInfo.principalAxes.axis1.y = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis1.y;
    parameters.cavityInfo.principalAxes.axis1.z = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis1.z;

    parameters.cavityInfo.principalAxes.axis2.x = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis2.x;
    parameters.cavityInfo.principalAxes.axis2.y = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis2.y;
    parameters.cavityInfo.principalAxes.axis2.z = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis2.z;

    parameters.cavityInfo.principalAxes.axis3.x = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis3.x;
    parameters.cavityInfo.principalAxes.axis3.y = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis3.y;
    parameters.cavityInfo.principalAxes.axis3.z = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.axis3.z;

    parameters.cavityInfo.principalAxes.moment1 = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.moment1;
    parameters.cavityInfo.principalAxes.moment2 = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.moment2;
    parameters.cavityInfo.principalAxes.moment3 = (CL_STRUCT_FLOAT)cavityInfo.principalAxes.moment3;
        // End Principal Axes
    parameters.cavityInfo.volume = (CL_STRUCT_FLOAT)cavityInfo.volume;
    parameters.cavityInfo.numCoords = (CL_STRUCT_INT)cavityInfo.numCoords;

    parameters.cavityInfo.gridStep.x = (CL_STRUCT_FLOAT)cavityInfo.gridStep.x;
    parameters.cavityInfo.gridStep.y = (CL_STRUCT_FLOAT)cavityInfo.gridStep.y;
    parameters.cavityInfo.gridStep.z = (CL_STRUCT_FLOAT)cavityInfo.gridStep.z;

    parameters.cavityInfo.minCoord.x = (CL_STRUCT_FLOAT)cavityInfo.minCoord.x;
    parameters.cavityInfo.minCoord.y = (CL_STRUCT_FLOAT)cavityInfo.minCoord.y;
    parameters.cavityInfo.minCoord.z = (CL_STRUCT_FLOAT)cavityInfo.minCoord.z;
    
    parameters.cavityInfo.maxCoord.x = (CL_STRUCT_FLOAT)cavityInfo.maxCoord.x;
    parameters.cavityInfo.maxCoord.y = (CL_STRUCT_FLOAT)cavityInfo.maxCoord.y;
    parameters.cavityInfo.maxCoord.z = (CL_STRUCT_FLOAT)cavityInfo.maxCoord.z;
    // End Cavity Info

    // Dihedral
    parameters.numDihedralElements = (CL_STRUCT_INT)numDihedralElements;
    
    // dockingSiteInfo
    double min[3];
    min[0] = parameters.cavityInfo.minCoord.x;
    min[1] = parameters.cavityInfo.minCoord.y;
    min[2] = parameters.cavityInfo.minCoord.z;
    double max[3];
    max[0] = parameters.cavityInfo.maxCoord.x;
    max[1] = parameters.cavityInfo.maxCoord.y;
    max[2] = parameters.cavityInfo.maxCoord.z;

    // Define box Receptor Atoms:
    parameters.dockingSiteInfo.minReceptor.x = min[0] - GOOD_RECEPTORS_BORDER;
    parameters.dockingSiteInfo.minReceptor.y = min[1] - GOOD_RECEPTORS_BORDER;
    parameters.dockingSiteInfo.minReceptor.z = min[2] - GOOD_RECEPTORS_BORDER;

    parameters.dockingSiteInfo.maxReceptor.x = max[0] + GOOD_RECEPTORS_BORDER;
    parameters.dockingSiteInfo.maxReceptor.y = max[1] + GOOD_RECEPTORS_BORDER;
    parameters.dockingSiteInfo.maxReceptor.z = max[2] + GOOD_RECEPTORS_BORDER;

    // Define box box Cavity:
    parameters.dockingSiteInfo.minCavity.x = min[0] - CAVITY_BORDER;
    parameters.dockingSiteInfo.minCavity.y = min[1] - CAVITY_BORDER;
    parameters.dockingSiteInfo.minCavity.z = min[2] - CAVITY_BORDER;

    parameters.dockingSiteInfo.maxCavity.x = max[0] + CAVITY_BORDER;
    parameters.dockingSiteInfo.maxCavity.y = max[1] + CAVITY_BORDER;
    parameters.dockingSiteInfo.maxCavity.z = max[2] + CAVITY_BORDER;

    // Define Grid:
    parameters.grid.gridStep[0] = (CL_STRUCT_FLOAT)gridProps.step.x;
    parameters.grid.gridStep[1] = (CL_STRUCT_FLOAT)gridProps.step.y;
    parameters.grid.gridStep[2] = (CL_STRUCT_FLOAT)gridProps.step.z;

    parameters.grid.gridMin[0] = (CL_STRUCT_FLOAT)std::floor(parameters.dockingSiteInfo.minCavity.x - 2 * parameters.grid.gridStep[0]);
    parameters.grid.gridMin[1] = (CL_STRUCT_FLOAT)std::floor(parameters.dockingSiteInfo.minCavity.y - 2 * parameters.grid.gridStep[1]);
    parameters.grid.gridMin[2] = (CL_STRUCT_FLOAT)std::floor(parameters.dockingSiteInfo.minCavity.z - 2 * parameters.grid.gridStep[2]);

    parameters.grid.gridMax[0] = (CL_STRUCT_FLOAT)std::ceil(parameters.dockingSiteInfo.maxCavity.x + 2 * parameters.grid.gridStep[0]);
    parameters.grid.gridMax[1] = (CL_STRUCT_FLOAT)std::ceil(parameters.dockingSiteInfo.maxCavity.y + 2 * parameters.grid.gridStep[1]);
    parameters.grid.gridMax[2] = (CL_STRUCT_FLOAT)std::ceil(parameters.dockingSiteInfo.maxCavity.z + 2 * parameters.grid.gridStep[2]);

    parameters.grid.NXYZ[0] = (CL_STRUCT_INT)( (parameters.grid.gridMax[0] - parameters.grid.gridMin[0]) / parameters.grid.gridStep[0]);
    parameters.grid.NXYZ[1] = (CL_STRUCT_INT)( (parameters.grid.gridMax[1] - parameters.grid.gridMin[1]) / parameters.grid.gridStep[1]);
    parameters.grid.NXYZ[2] = (CL_STRUCT_INT)( (parameters.grid.gridMax[2] - parameters.grid.gridMin[2]) / parameters.grid.gridStep[2]);

    parameters.grid.N = (CL_STRUCT_INT)(parameters.grid.NXYZ[0] * parameters.grid.NXYZ[1] * parameters.grid.NXYZ[2]);
    parameters.grid.totalN = (CL_STRUCT_INT)(parameters.grid.N * TOTAL_PLP_CLASSES);

    parameters.grid.SXYZ[0] = (CL_STRUCT_INT)(parameters.grid.NXYZ[1] * parameters.grid.NXYZ[2]);
    parameters.grid.SXYZ[1] = (CL_STRUCT_INT)parameters.grid.NXYZ[2];
    parameters.grid.SXYZ[2] = (CL_STRUCT_INT)1;

    parameters.grid.totalPLPClasses = (CL_STRUCT_INT)TOTAL_PLP_CLASSES;

    gridSize = sizeof(GridPointGPU) * parameters.grid.N;
}

void Data::initPopulations(double** populationsFromFile) {

    globalPopulations = new CL_STRUCT_FLOAT[parameters.nruns * parameters.popMaxSize * parameters.chromStoreLen];
    globalPopulationsSize = sizeof(CL_STRUCT_FLOAT) * parameters.nruns * parameters.popMaxSize * parameters.chromStoreLen;
    globalPopulationsCopy = new CL_STRUCT_FLOAT[parameters.nruns * parameters.popMaxSize * parameters.chromStoreLen];
    globalPopulationsCopySize = sizeof(CL_STRUCT_FLOAT) * parameters.nruns * parameters.popMaxSize * parameters.chromStoreLen;
    // Convert to 1D array for GPU
    for (int i = 0; i < parameters.nruns; i++) {
        for (int j = 0; j < parameters.popSize; j++) {
            for (int k = 0; k < parameters.chromStoreLen; k++) {
                globalPopulations[i*parameters.popMaxSize*parameters.chromStoreLen+j*parameters.chromStoreLen+k] = (CL_STRUCT_FLOAT)populationsFromFile[i*parameters.popSize+j][k];
            }
        }
    }
}

void Data::initAtoms(Atom* atomsFromFile, AtomGPU* atomsGPU, uint32_t numAtoms) {

    for (int i = 0; i < numAtoms; i++) {
        atomsGPU[i].id = (CL_STRUCT_INT)(atomsFromFile[i].id);
        atomsGPU[i].atomicNo = (CL_STRUCT_INT)(atomsFromFile[i].atomicNo);
        atomsGPU[i].numImplicitHydrogens = (CL_STRUCT_INT)(atomsFromFile[i].numImplicitHydrogens);
        atomsGPU[i].triposType = (CL_STRUCT_INT)(atomsFromFile[i].triposType);
        atomsGPU[i].atomicMass = (CL_STRUCT_FLOAT)(atomsFromFile[i].atomicMass);
        atomsGPU[i].formalCharge = (CL_STRUCT_FLOAT)(atomsFromFile[i].formalCharge);
        atomsGPU[i].partialCharge = (CL_STRUCT_FLOAT)(atomsFromFile[i].partialCharge);
        atomsGPU[i].groupCharge = (CL_STRUCT_FLOAT)(atomsFromFile[i].groupCharge);
        atomsGPU[i].x = (CL_STRUCT_FLOAT)(atomsFromFile[i].x);
        atomsGPU[i].y = (CL_STRUCT_FLOAT)(atomsFromFile[i].y);
        atomsGPU[i].z = (CL_STRUCT_FLOAT)(atomsFromFile[i].z);
        atomsGPU[i].vdwRadius = (CL_STRUCT_FLOAT)(atomsFromFile[i].vdwRadius);
        atomsGPU[i].cyclicFlag = (CL_STRUCT_INT)(atomsFromFile[i].cyclicFlag);
        atomsGPU[i].enabled = (CL_STRUCT_INT)(atomsFromFile[i].enabled);
    }
}

void Data::initLigandAtoms(Atom* ligandAtomsFromFile) {

    ligandAtoms = new AtomGPU[parameters.ligandNumAtoms];
    ligandAtomsSize = sizeof(AtomGPU) * parameters.ligandNumAtoms;
    initAtoms(ligandAtomsFromFile, ligandAtoms, parameters.ligandNumAtoms);
}

void Data::initReceptorAtoms(Atom* receptorAtomsFromFile) {
    receptorAtoms = new AtomGPU[parameters.receptorNumAtoms];
    receptorAtomsSize = sizeof(AtomGPU) * parameters.receptorNumAtoms;
    initAtoms(receptorAtomsFromFile, receptorAtoms, parameters.receptorNumAtoms);
}

void Data::initBonds(Bond* bondsFromFile, BondGPU* bondsGPU, uint32_t numBonds) {

    for (int i = 0; i < numBonds; i++) {
        bondsGPU[i].id = (CL_STRUCT_INT)(bondsFromFile[i].id);
        bondsGPU[i].atom1ID = (CL_STRUCT_INT)(bondsFromFile[i].atom1ID);
        bondsGPU[i].atom2ID = (CL_STRUCT_INT)(bondsFromFile[i].atom2ID);
        bondsGPU[i].formalBondOrder = (CL_STRUCT_INT)(bondsFromFile[i].formalBondOrder);
        bondsGPU[i].partialBondOrder = (CL_STRUCT_FLOAT)(bondsFromFile[i].partialBondOrder);
        bondsGPU[i].cyclicFlag = (CL_STRUCT_INT)(bondsFromFile[i].cyclicFlag);
        bondsGPU[i].isRotatable = (CL_STRUCT_INT)(bondsFromFile[i].isRotatable);
    }
}

void Data::initLigandBonds(Bond* ligandBondsFromFile) {

    ligandBonds = new BondGPU[parameters.ligandNumBonds];
    ligandBondsSize = sizeof(BondGPU) * parameters.ligandNumBonds;
    initBonds(ligandBondsFromFile, ligandBonds, parameters.ligandNumBonds);
}

void Data::initDihedralRefData(DihedralRefData* dihedralRefDataFromFile) {

    dihedralRefData = new DihedralRefDataGPU[parameters.numDihedralElements];
    dihedralRefDataSize = sizeof(DihedralRefDataGPU) * parameters.numDihedralElements;

    for (int i = 0; i < parameters.numDihedralElements; i++) {
        dihedralRefData[i].bondID = (CL_STRUCT_INT)dihedralRefDataFromFile[i].header.bondID;
        dihedralRefData[i].atom1ID = (CL_STRUCT_INT)dihedralRefDataFromFile[i].header.atom1ID;
        dihedralRefData[i].atom2ID = (CL_STRUCT_INT)dihedralRefDataFromFile[i].header.atom2ID;
        dihedralRefData[i].atom3ID = (CL_STRUCT_INT)dihedralRefDataFromFile[i].header.atom3ID;
        dihedralRefData[i].atom4ID = (CL_STRUCT_INT)dihedralRefDataFromFile[i].header.atom4ID;
        dihedralRefData[i].numRotAtoms = (CL_STRUCT_INT)dihedralRefDataFromFile[i].header.numRotAtoms;
        dihedralRefData[i].stepSize = (CL_STRUCT_FLOAT)dihedralRefDataFromFile[i].header.stepSize;

        if (dihedralRefData[i].numRotAtoms < 200) {
            for (int j = 0; j < dihedralRefData[i].numRotAtoms; j++) {
                dihedralRefData[i].rotAtomsIDs[j] = (CL_STRUCT_INT)dihedralRefDataFromFile[i].rotAtomsIDs[j];
            }
        } else {
            dbglWE(__FILE__, __FUNCTION__, __LINE__, "Static structure DihedralRefDataGPU.rotAtomsIDs too small, increase size 200->" + std::to_string(dihedralRefData[i].numRotAtoms) + ".");
        }
    }
}

void Data::initSeed() {

    seed = new cl_ulong[parameters.maxThreads];
    seedSize = sizeof(cl_ulong) * parameters.maxThreads;

    srand((unsigned) time(NULL));
	for (int i = 0; i < parameters.maxThreads; i++) {
		seed[i] = rand();
	}
}

inline int index2D(int l, int x, int y) {
    return l * x + y;
}

void Data::initLigandAtomPairsForClash() {
    const int maxPathLen = INT_MAX / 2; // "/ 2": so we can add to it.
    int l = parameters.ligandNumAtoms;
    int x,y;
    
    // 2D arrays:
    int* connectionsTable = new int[l * l]();// Init to zero!
    int* shortestPathTable = new int[l * l];
    short* isDihedralTable = new short[l * l]();// Init to zero!

    // Set entire 2D array to maxPathLen
    for (int i = 0; i < l * l; i++) {
        shortestPathTable[i] = maxPathLen;
    }
    // Set [i][i] to 0 (each to oneself distance = 0)
    for (int i = 0; i < l; i++) {
        shortestPathTable[i * l + i] = 0;
    }
    // fill connection table with bonds
    for (int i = 0; i < parameters.ligandNumBonds; i++) {
        x = ligandBonds[i].atom1ID - 1;
        y = ligandBonds[i].atom2ID - 1;

        connectionsTable[index2D(l, x, y)] = 2;
        connectionsTable[index2D(l, y, x)] = 2;
    }
    // overlay dihedral bonds
    for (int i = 0; i < parameters.numDihedralElements; i++) {
        int bondID = dihedralRefData[i].bondID;
        int bondIndex = bondID - 1;
        if(ligandBonds[bondIndex].id == bondID) {
            x = ligandBonds[bondIndex].atom1ID - 1;
            y = ligandBonds[bondIndex].atom2ID - 1;

            connectionsTable[index2D(l, x, y)] = 1;
            connectionsTable[index2D(l, y, x)] = 1;
        } else {
            dbglWE(__FILE__, __FUNCTION__, __LINE__, "Bond ID("+std::to_string(ligandBonds[bondIndex].id)+") and index("+std::to_string(bondIndex)+") MISMATCH]");
        }
    }

    // FIXME: improve graph algorithm

    // for every atom, except last
    for (int a = 0; a < l - 1; a++) {
        // iteration (numAt - 1)
        for (int i = 0; i < l - 1; i++) {
            // for every atom (a = a2 is a starting point in the 1st iteration)
            for (int a2 = 0; a2 < l; a2++) {
                // if we can reach
                if (shortestPathTable[index2D(l, a, a2)] != maxPathLen) {
                    // for atom connection > a
                    for (int c = a + 1; c < l; c++) {
                        // for main atom a
                        // iteration i
                        // current atom a2
                        // connection c

                        // there is bond
                        if (connectionsTable[index2D(l, a2, c)] > 0) {

                            // Calculate path length and check if shorter
                            int pathLen = shortestPathTable[index2D(l, a, a2)] + 1;
                            if (pathLen < shortestPathTable[index2D(l, a, c)]) {
                                shortestPathTable[index2D(l, a, c)] = pathLen;
                                shortestPathTable[index2D(l, c, a)] = pathLen;

                                // if Dihedral bond then mark it, otherwise reset it.
                                if (connectionsTable[index2D(l, a2, c)] == 1 || // current bond is dihedral
                                    isDihedralTable[index2D(l, a, a2)] == 1) {   // there was a dihedral bond at some point

                                    isDihedralTable[index2D(l, a, c)] = 1;
                                    isDihedralTable[index2D(l, c, a)] = 1;
                                } else {
                                    isDihedralTable[index2D(l, a, c)] = 0;
                                    isDihedralTable[index2D(l, c, a)] = 0;
                                }
                            } 
                        }
                    }
                }
            }
        }
    }

    AtomPairIndex* atomPairIndexes = new AtomPairIndex[l * (l-1) / 2];// max. num. of pairs.

    // now check pairs to add to, only above diagonal
    int numPairs = 0;
    for (int i = 0; i < l; i++) {
        for (int j = i + 1; j < l; j++) {

            if (shortestPathTable[index2D(l, i, j)] == maxPathLen) {
                dbglWE(__FILE__, __FUNCTION__, __LINE__, "Atom with index: ("+std::to_string(i)+") and index("+std::to_string(j)+") not reachable in graph aka. path=maxPathLen (atomID = index + 1), possible error.");
            }
            
            // if path > 2, there is at least one dihedral and no light atom (aka. Hydrogen), add pair
            if (shortestPathTable[index2D(l, i, j)] > 2 &&
                isDihedralTable[index2D(l, i, j)] == 1 &&
                ligandAtoms[i].triposType != TRIPOS_TYPE_H && ligandAtoms[i].triposType != TRIPOS_TYPE_H_P &&
                ligandAtoms[j].triposType != TRIPOS_TYPE_H && ligandAtoms[j].triposType != TRIPOS_TYPE_H_P) {
                
                atomPairIndexes[numPairs].i = i;
                atomPairIndexes[numPairs].j = j;
                numPairs++;
            }
        }
    }

    parameters.numLigandAtomPairsForClash = numPairs;
    parameters.numUniqueAtom1PairsForClash = 0;
    int numPairsCheck = 1;
    if (numPairs == 0) {
        numPairs = 1;
        numPairsCheck = 0;
    }
    ligandAtomPairsForClash = new LigandAtomPairsForClash[numPairs];
    ligandAtomPairsForClashSize = sizeof(LigandAtomPairsForClash) * numPairs;

    if (numPairsCheck == 1) {
        for (int i = 0; i < numPairs; i++) {
            ligandAtomPairsForClash[i].numBondsBetween = shortestPathTable[index2D(l, atomPairIndexes[i].i, atomPairIndexes[i].j)];
            ligandAtomPairsForClash[i].atom1ID = atomPairIndexes[i].i + 1;
            ligandAtomPairsForClash[i].atom2ID = atomPairIndexes[i].j + 1;
        }

        // Count so that Atom1 can be accessed only once when calculating PLPclash.
        int currentAtom = 0;
        int numUniqueAtom1 = 0;
        while (currentAtom < numPairs) {
            
            int atom1 = currentAtom;
            int atom1ID = ligandAtomPairsForClash[currentAtom].atom1ID;
            int numAtom1 = 0;
            while(ligandAtomPairsForClash[currentAtom + numAtom1].atom1ID == atom1ID) {
                numAtom1++;
            }
            for(int a = currentAtom; a < currentAtom + numAtom1; a++) {
                ligandAtomPairsForClash[a].numAtom1 = numAtom1;
            }
            currentAtom += numAtom1;
            numUniqueAtom1++;
        }
        parameters.numUniqueAtom1PairsForClash = numUniqueAtom1;
    }

    delete[] connectionsTable;
    delete[] shortestPathTable;
    delete[] isDihedralTable;
    delete[] atomPairIndexes;
}

Data::Data(std::string file, Batch& batchRef) : batch(batchRef) {

    TIMER_START(t_dataPrep);

    FILE* File;
    openFileBinary(file, File);

    // Read the file header
    Header header;
    fread(&header, sizeof(Header), 1, File);

    // Read the ligand data (atoms, bonds)
    LigandFlex ligandFlex;
    fread(&ligandFlex, sizeof(LigandFlex), 1, File);

    auto ligandAtomsFromFile = new Atom[header.ligandNumAtoms];
    fread(ligandAtomsFromFile, sizeof(Atom), header.ligandNumAtoms, File);

    auto ligandBondsFromFile = new Bond[header.ligandNumBonds];
    fread(ligandBondsFromFile, sizeof(Bond), header.ligandNumBonds, File);

    // Ligand - dihedral data (Dihedral elements correspond to the first genes in a chromosome)

    // Number of dihedral elements (chromStoreLen - 6[com,orient] -2[score,RWfitness])
    uint32_t numDihedralElements = (header.chromStoreLen) - 6 - 2;
    
    auto dihedralRefDataFromFile = new DihedralRefData[numDihedralElements];

    for(int i=0; i < numDihedralElements; i++){
        // Read header
        fread(&(dihedralRefDataFromFile[i].header), sizeof(DihedralRefDataHeader), 1, File);
        // Reserve space for rotAtomsIDs
        uint32_t numRotAtoms = dihedralRefDataFromFile[i].header.numRotAtoms;
        dihedralRefDataFromFile[i].rotAtomsIDs = new uint32_t[numRotAtoms];
        // Read rotAtomsIDs
        fread(dihedralRefDataFromFile[i].rotAtomsIDs, sizeof(uint32_t), numRotAtoms, File);
    }

    // Read the receptor data (atoms, bonds)
    auto receptorAtomsFromFile = new Atom[header.receptorNumAtoms];
    fread(receptorAtomsFromFile, sizeof(Atom), header.receptorNumAtoms, File);

    auto receptorBonds = new Bond[header.receptorNumBonds];
    fread(receptorBonds, sizeof(Bond), header.receptorNumBonds, File);

    // GA parameters
    auto gaParams = new GAParams[header.numGAruns];
    fread(gaParams, sizeof(GAParams), header.numGAruns, File);

    // Grid (read outputs of RbtBaseGrid::OwnWrite and RbtRealGrid::OwnWrite)

    // Skip title (RbtBaseGrid)
    uint32_t titleLength;
    fread(&titleLength, sizeof(titleLength), 1, File);
    fseek(File,titleLength,SEEK_CUR);
    // Read grid properties
    GridProps gridProps;
    fread(&gridProps, sizeof(GridProps), 1, File);
    // Skip title (RbtRealGrid)
    fread(&titleLength, sizeof(titleLength), 1, File);
    fseek(File,titleLength,SEEK_CUR);
    // Read tolerance for comparing grid values
    double grid_tol;
    fread(&grid_tol,sizeof(double),1,File);
    // Read grid data (RbtRealGrid)
    // Grid is actually a tensor or 3-D array, accessed as grid(i, j, k), indicies from 0
    // m_grid = Eigen::Tensor<float, 3, Eigen::RowMajor>(nX, nY, nZ);
    // We store the grid in an array (row-major ordering)
    // Dimensions of a grid are [NX x NY x NZ]
    auto gridRealData = new float[gridProps.N];
    fread(gridRealData, sizeof(float), gridProps.N, File);

    // Cavity (a list of points at active site of a protein)
    CavityInfo cavityInfo;
    fread(&cavityInfo, sizeof(CavityInfo), 1, File);
    auto cavityCoords = new Coord[cavityInfo.numCoords];
    fread(cavityCoords, sizeof(Coord), cavityInfo.numCoords, File);

    // Read the initial populations over multiple nruns

    // Each population block consists of the following:
    // - population mean score (double)
    // - population score variance (double)
    // - chromosomes (popSize * chromStoreLen doubles)
    //
    // Each "chromosome" consists of actual chromosome data
    // (dihedral angles for each rotatable bond, COM, orientation)      COM = Center Of Mass
    // plus two doubles: score and RWFitness
    // | bond 1 | ... | bond B | COM | orientation | score | RWFitness |
    // |---1----| ... |---1----|--3--|------3------|---1---|-----1-----|

    // Reserve space
    // We will store score mean and variance in separate arrays,
    // and chromosome data in another 2D array of
    // (nruns * popSize) rows and chromStoreLen columns.
    auto popScoreMean = new double[header.nruns];
    auto popScoreVariance = new double[header.nruns];
    auto populationsFromFile = new double*[header.nruns * header.popSize];
    for (int i = 0; i < header.nruns * header.popSize; i++) {
        populationsFromFile[i] = new double[header.chromStoreLen];
    }
    // Read population blocks
    for (int i = 0; i < header.nruns; i++) {
        fread(popScoreMean + i, sizeof(double), 1, File);
        fread(popScoreVariance + i, sizeof(double), 1, File);
        for (int j = 0; j < header.popSize; j++) {
            fread(populationsFromFile[i * header.popSize + j], sizeof(double), header.chromStoreLen, File);
        }
    }

    fclose(File);

    // Convert data to OpenCL

    initParameres(header, ligandFlex, gaParams[0], cavityInfo, numDihedralElements, gridProps);
    initPopulations(populationsFromFile);
    initLigandAtoms(ligandAtomsFromFile);
    initReceptorAtoms(receptorAtomsFromFile);
    initLigandBonds(ligandBondsFromFile);
    ligandAtomsSmallGlobalAll = new AtomGPUsmall[parameters.nruns * parameters.popMaxSize * parameters.ligandNumAtoms];
    ligandAtomsSmallGlobalAllSize = sizeof(AtomGPUsmall) * parameters.nruns * parameters.popMaxSize * parameters.ligandNumAtoms;
    initDihedralRefData(dihedralRefDataFromFile);
    equalsArray = new CL_STRUCT_INT[parameters.nruns * parameters.nReplicates];
    equalsArraySize = sizeof(CL_STRUCT_INT) * parameters.nruns * parameters.nReplicates;
    receptorIndex = new CL_STRUCT_INT[parameters.receptorNumAtoms];
    receptorIndexSize = sizeof(CL_STRUCT_INT) * parameters.receptorNumAtoms;
    bestScore = new CL_STRUCT_FLOAT[parameters.nruns];
    bestScoreSize = sizeof(CL_STRUCT_FLOAT) * parameters.nruns;
    score = new CL_STRUCT_FLOAT[parameters.nruns];
    scoreSize = sizeof(CL_STRUCT_FLOAT) * parameters.nruns;
    popNewIndex = new CL_STRUCT_INT[2];
    popNewIndexSize = sizeof(CL_STRUCT_INT) * 2;
    popNewIndex[0] = (CL_STRUCT_INT)0;
    popNewIndex[1] = parameters.popSize;
    initSeed();
    initLigandAtomPairsForClash();
    rngStates = new tyche_i_state[parameters.maxThreads];
    rngStatesSize = sizeof(tyche_i_state) * parameters.maxThreads;
    numGoodReceptors = (CL_STRUCT_INT)0;
    numGoodReceptorsSize = sizeof(CL_STRUCT_INT);

    // Delete Original Arrays
    delete[] ligandAtomsFromFile;
    delete[] ligandBondsFromFile;

    for(int i=0; i < numDihedralElements; i++){
        delete[] dihedralRefDataFromFile[i].rotAtomsIDs;
    }
    delete[] dihedralRefDataFromFile;

    delete[] receptorAtomsFromFile;
    delete[] receptorBonds;

    delete[] gaParams;
    delete[] gridRealData;
    delete[] cavityCoords;

    delete[] popScoreMean;
    delete[] popScoreVariance;
    for (int i = 0; i < header.nruns * header.popSize; i++) {
        delete[] populationsFromFile[i];
    }
    delete[] populationsFromFile;

    TIMER_END(t_dataPrep, tot_dataPrep);
}

Data::~Data() {

    // Write result file
    saveResultsToFile(batch.outputPath + "/", batch.outputOnlyBestN);

    // Write timers
    if(batch.timeKernels == 1) {
        saveTimersToFile(batch.outputPath + "/");
    }

    // Write Scores
    if(batch.trackScores == 1) {
        saveScoresToFile(batch.outputPath + "/");
        delete[] scoreTracker;
    }

    delete[] globalPopulations;
    delete[] globalPopulationsCopy;
    delete[] ligandAtoms;
    delete[] receptorAtoms;
    delete[] ligandBonds;
    delete[] ligandAtomsSmallGlobalAll;
    delete[] dihedralRefData;
    delete[] equalsArray;
    delete[] receptorIndex;
    delete[] bestScore;
    delete[] score;
    delete[] popNewIndex;
    delete[] seed;
    delete[] ligandAtomPairsForClash;
    delete[] rngStates;
}

void Data::saveResultsToFile(std::string path, int outputOnlyBestN) {

    // get current date:
    time_t curtime;
    time(&curtime);
    std::string fileName = std::string(ctime(&curtime));
    std::replace(fileName.begin(), fileName.end(), ':', '_');
    std::replace(fileName.begin(), fileName.end(), '\n', '_');

    std::string completeFilePath = path + fileName + ".sdf";

    // open file
    FILE *fout;
    openFileC(completeFilePath, fout);

    char elements[118][4] = {"H  ", "He ", "Li ", "Be ", "B  ", "C  ", "N  ", "O  ", "F  ",
                             "Ne ", "Na ", "Mg ", "Al ", "Si ", "P  ", "S  ", "Cl ", "Ar ",
                             "K  ", "Ca ", "Sc ", "Ti ", "V  ", "Cr ", "Mn ", "Fe ", "Co ",
                             "Ni ", "Cu ", "Zn ", "Ga ", "Ge ", "As ", "Se ", "Br ", "Kr ",
                             "Rb ", "Sr ", "Y  ", "Zr ", "Nb ", "Mo ", "Tc ", "Ru ", "Rh ",
                             "Pd ", "Ag ", "Cd ", "In ", "Sn ", "Sb ", "Te ", "I  ", "Xe ",
                             "Cs ", "Ba ", "La ", "Ce ", "Pr ", "Nd ", "Pm ", "Sm ", "Eu ",
                             "Gd ", "Tb ", "Dy ", "Ho ", "Er ", "Tm ", "Yb ", "Lu ", "Hf ",
                             "Ta ", "W  ", "Re ", "Os ", "Ir ", "Pt ", "Au ", "Hg ", "Tl ",
                             "Pb ", "Bi ", "Po ", "At ", "Rn ", "Fr ", "Ra ", "Ac ", "Th ",
                             "Pa ", "U  ", "Np ", "Pu ", "Am ", "Cm ", "Bk ", "Cf ", "Es ",
                             "Fm ", "Md ", "No ", "Lr ", "Rf ", "Db ", "Sg ", "Bh ", "Hs ",
                             "Mt ", "Ds ", "Rg ", "Cn ", "Nh ", "Fl ", "Mc ", "Lv ", "Ts ",
                             "Og "};

    // Sort
    int* iSorted = new int[parameters.nruns];
    for (int i = 0; i < parameters.nruns; i++) {
        iSorted[i] = i;
    }

    int outLength = parameters.nruns;

    if(outputOnlyBestN != 0) {

        outLength = outputOnlyBestN;

        float tempRW;
        int tempI;
        for (int i = 0; i < parameters.nruns-1; i++) {
            for (int j = 0; j < parameters.nruns-i-1; j++) {
                if (score[j] > score[j+1]) {
                    // Swap Values
                    tempRW = score[j];
                    score[j] = score[j+1];
                    score[j+1] = tempRW;
                    // Swap Indexes
                    tempI = iSorted[j];
                    iSorted[j] = iSorted[j+1];
                    iSorted[j+1] = tempI;
                }
            }
        }
    }

    // For all runs
    for (int i = 0; i < parameters.nruns && i < outLength; i++) {

        int sortedI = iSorted[i];

        // Header (3 lines)
        fprintf(fout,"header:name\nheader:software\nheader:notes\n");

        // Counts line
        fprintf(fout,"%3d%3d  0  0  1  0            999 V2000\n", parameters.ligandNumAtoms, parameters.ligandNumBonds);

        // Atoms block
        for(int j=0; j < parameters.ligandNumAtoms; j++){
            int offset = sortedI * parameters.ligandNumAtoms;
            fprintf(fout, "%10.4f%10.4f%10.4f %3s 0  0  0  0  0  0\n",ligandAtomsSmallGlobalAll[offset+j].x, ligandAtomsSmallGlobalAll[offset+j].y, ligandAtomsSmallGlobalAll[offset+j].z, elements[ligandAtomsSmallGlobalAll[offset+j].atomicNo-1]);
        }

        // Bonds block
        for(int j=0; j < parameters.ligandNumBonds; j++){
            fprintf(fout,"%3d%3d%3d  0  0  0\n",ligandBonds[j].atom1ID, ligandBonds[j].atom2ID, ligandBonds[j].formalBondOrder);
        }
        
        // End of properties block
        fprintf(fout,"M  END\n");

        // Custom data - score
        fprintf(fout,"> <PLP_SCORE>\n%f\n", score[i]); // FIXME: Change to bestScore when implemented

        // End of ligand
        fprintf(fout,"\n$$$$\n");

    }

    // close
    fclose(fout);

    delete[] iSorted;
}

void Data::saveTimersToFile(std::string path) {

    // get current date:
    time_t curtime;
    time(&curtime);
    std::string fileName = std::string(ctime(&curtime));
    std::replace(fileName.begin(), fileName.end(), ':', '_');
    std::replace(fileName.begin(), fileName.end(), '\n', '_');

    std::string completeFilePath = path + fileName + "_timers.csv";

    // open file
    FILE *fout;
    openFileC(completeFilePath, fout);

    fprintf(fout,"Data preparation time,%lf\r\n", tot_dataPrep);
    fprintf(fout,"WorkerCLCreation time,%lf\r\n", tot_workerCreation);
    fprintf(fout,"Kernel creation time,%lf\r\n", tot_kernelCreation);
    fprintf(fout,"Data to GPU time,%lf\r\n", tot_dataToGPU);
    fprintf(fout,"Data to CPU time,%lf\r\n", tot_dataToCPU);
    fprintf(fout,"Kernels set args time,%lf\r\n", tot_kernelSetArgs);
    double totalKernelTime = tot_kernelInit + tot_kernelInit2 + tot_kernelInit3 + tot_kernelInitGrid
                            + tot_kernelSyncToModel + tot_kernelPLP
                            + tot_kernelSort + tot_kernelNormalize
                            + tot_kernelCreateNew
                            + tot_kernelFinalize;// +...
    fprintf(fout,"Total Kernel run time,%lf\r\n", totalKernelTime);
    fprintf(fout,"kernelInit time,%lf\r\n", tot_kernelInit);
    fprintf(fout,"kernelInit2 time,%lf\r\n", tot_kernelInit2);
    fprintf(fout,"kernelInit3 time,%lf\r\n", tot_kernelInit3);
    fprintf(fout,"kernelInitGrid time,%lf\r\n", tot_kernelInitGrid);
    fprintf(fout,"kernelSyncToModel time,%lf\r\n", tot_kernelSyncToModel);
    fprintf(fout,"kernelPLP time,%lf\r\n", tot_kernelPLP);
    fprintf(fout,"kernelSort time,%lf\r\n", tot_kernelSort);
    fprintf(fout,"kernelNormalize time,%lf\r\n", tot_kernelNormalize);
    fprintf(fout,"kernelCreateNew time,%lf\r\n", tot_kernelCreateNew);
    fprintf(fout,"kernelFinalize time,%lf\r\n", tot_kernelFinalize);

    // close
    fclose(fout);
}

void Data::saveScoresToFile(std::string path) {

    // get current date:
    time_t curtime;
    time(&curtime);
    std::string fileName = std::string(ctime(&curtime));
    std::replace(fileName.begin(), fileName.end(), ':', '_');
    std::replace(fileName.begin(), fileName.end(), '\n', '_');

    std::string completeFilePath = path + fileName + "_scores.csv";

    // open file
    FILE *fout;
    openFileC(completeFilePath, fout);

    // 1. print Header:
    fprintf(fout,"GA_step");
    for(int i = 0; i < parameters.nruns; i++) {
        fprintf(fout,",%d.run", i);
    }
    fprintf(fout,"\r\n");

    // 2. print GA_step Scores
    for (int i = 0; i < numOfRunsDone; i++) {

        // 2.a print GA_step Number
        fprintf(fout,"%d", i);

        // 2.b print Scores
        int baseIndex = i * parameters.nruns;

        for(int j = baseIndex; j < baseIndex + parameters.nruns; j++){
            fprintf(fout,",%f", scoreTracker[j]);
        }

        fprintf(fout,"\r\n");
    }

    // close
    fclose(fout);
}

void Data::trackScore() {

    CL_STRUCT_FLOAT* tempScore;

    if(numOfRunsDone == 0) {
        scoreTracker = new CL_STRUCT_FLOAT[parameters.nruns * (parameters.ncycles + 1)];
        tempScore = bestScore;
    } else {
        tempScore = score;
    }
    
    uint32_t baseIndex = numOfRunsDone * parameters.nruns;

    for(int i = baseIndex; i < baseIndex + parameters.nruns; i++) {
        scoreTracker[i] = tempScore[i - baseIndex];
    }

    numOfRunsDone++;
}
