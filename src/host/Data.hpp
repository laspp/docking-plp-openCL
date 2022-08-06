#pragma once

#include <string>
#include <vector>
#include "../kernels/clStructs.h"
#include "Batch.hpp"

typedef struct Header_t {
    uint32_t ligandNumAtoms;
    uint32_t ligandNumBonds;
    uint32_t ligandStaticDataLen;
    uint32_t receptorNumAtoms;
    uint32_t receptorNumBonds;
    uint32_t receptorStaticDataLen;
    uint32_t nruns;
    uint32_t popSize;
    uint32_t popMaxSize;
    uint32_t popStaticDataLen;
    uint32_t chromStoreLen;
    uint32_t numGAruns;
} Header;

typedef struct LigandFlex_t {
    double dihedral_step;
    double rot_step;
    double trans_step;
    double max_dihedral;
    double max_rot;
    double max_trans;
    uint8_t transMode;
    uint8_t rotMode;
    uint8_t dihedralMode;
} LigandFlex;

typedef struct Atom_t {
    uint32_t id;
    uint32_t atomicNo;
    uint32_t numImplicitHydrogens;
    uint32_t triposType;// classified atom
    double atomicMass;
    double formalCharge;
    double partialCharge;
    double groupCharge;
    double x;
    double y;
    double z;
    double vdwRadius;
    uint8_t cyclicFlag;
    uint8_t enabled;
} Atom;

typedef struct Bond_t {
    uint32_t id;
    uint32_t atom1ID;
    uint32_t atom2ID;
    uint32_t formalBondOrder;// bond type (single, double, tripple bond)
    double partialBondOrder;
    uint8_t cyclicFlag;
    uint8_t isRotatable;
} Bond;

typedef struct DihedralRefDataHeader_t {
    uint32_t bondID;        // ID of a bond that corresponds to the dihedral element
    uint32_t atom1ID;
    uint32_t atom2ID;
    uint32_t atom3ID;
    uint32_t atom4ID;
    uint32_t numRotAtoms;   // number of rotatable atoms
    double stepSize;        // dihedral step size - same as dihedral_step in LigandFlexData
} DihedralRefDataHeader;

typedef struct DihedralRefData_t {
    DihedralRefDataHeader header;
    uint32_t* rotAtomsIDs;  // array of rotatable atoms IDs
} DihedralRefData;

typedef struct GAParams_t {
    uint32_t history_freq;
    uint32_t nconvergence;
    uint32_t ncycles;
    double  equality_threshold;
    double  new_fraction;
    double pcrossover;
    double step_size;
    uint8_t cmutate;
    uint8_t  xovermut;
} GAParams;

typedef struct Coord_t {
    double x;
    double y;
    double z;
} Coord;

typedef struct GridProps_t {
    Coord min;        // Minimum grid coords (real-world units)
    Coord max;        // Maximum grid coords (real-world units)
    Coord step;       // Grid step in X,Y,Z (real-world units)
    Coord padMin;     // Minimum pad coords (used to define an active region of the grid)
    Coord padMax;     // Maximum pad coords (used to define an active region of the grid)
    uint32_t NX;      // Number of grid points in X direction (fixed in constructor)
    uint32_t NY;      // Number of grid points in Y direction (fixed in constructor)
    uint32_t NZ;      // Number of grid points in Z direction (fixed in constructor)
    uint32_t N;       // Total number of grid points (fixed in constructor)
    uint32_t SX;      // Stride of X
    uint32_t SY;      // Stride of Y
    uint32_t SZ;      // Stride of Z
    uint32_t NPad;    // Defines a zero-padded border around the grid
    int32_t nXMin;    // Minimum X grid point (in units of grid step)
    int32_t nXMax;    // Maximum X grid point (in units of grid step)
    int32_t nYMin;    // Minimum Y grid point (in units of grid step)
    int32_t nYMax;    // Maximum Y grid point (in units of grid step)
    int32_t nZMin;    // Minimum Z grid point (in units of grid step)
    int32_t nZMax;    // Maximum Z grid point (in units of grid step)
} GridProps;

typedef struct PrincipalAxes_t {
    Coord center;
    Coord axis1;
    Coord axis2;
    Coord axis3;
    double moment1;
    double moment2;
    double moment3;
} PrincipalAxes;

typedef struct CavityInfo_t {
    PrincipalAxes principalAxes;
    double volume;
    uint32_t numCoords;
    Coord gridStep;
    Coord minCoord;
    Coord maxCoord;
} CavityInfo;

typedef union{
	struct{
		cl_uint a,b,c,d;
	};
	cl_ulong res;
} tyche_i_state;

typedef struct AtomPairIndex_t {
    int i;
    int j;
} AtomPairIndex;

class Data {

   void initParameres(Header& header, LigandFlex& ligandFlex, GAParams& gaParams, CavityInfo& cavityInfo, uint32_t numDihedralElements, GridProps& gridProps);
   void initPopulations(double** populationsFromFile);
   void initAtoms(Atom* atomsFromFile, AtomGPU* atomsGPU, uint32_t numAtoms);
   void initLigandAtoms(Atom* ligandAtomsFromFile);
   void initReceptorAtoms(Atom* receptorAtomsFromFile);
   void initBonds(Bond* bondsFromFile, BondGPU* bondsGPU, uint32_t numBonds);
   void initLigandBonds(Bond* ligandBondsFromFile);
   void initDihedralRefData(DihedralRefData* dihedralRefDataFromFile);
   void initSeed();
   void initLigandAtomPairsForClash();

    void saveResultsToFile(std::string path, int outputOnlyBestN);
    void saveTimersToFile(std::string path);
    void saveScoresToFile(std::string path);

    Batch& batch;

    // Score:
    uint32_t numOfRunsDone = 0;
    CL_STRUCT_FLOAT* scoreTracker;

public:

    const uint32_t LOCAL_SIZE = 256;

    // Stack:
    parametersForGPU parameters;
    uint32_t parametersSize;

    CL_STRUCT_INT numGoodReceptors;
    uint32_t numGoodReceptorsSize;

    // Allocated:
    CL_STRUCT_FLOAT* globalPopulations;
    uint32_t globalPopulationsSize;
    CL_STRUCT_FLOAT* globalPopulationsCopy;
    uint32_t globalPopulationsCopySize;

    AtomGPU* ligandAtoms;
    uint32_t ligandAtomsSize;
    AtomGPU* receptorAtoms;
    uint32_t receptorAtomsSize;

    BondGPU* ligandBonds;
    uint32_t ligandBondsSize;

    AtomGPUsmall* ligandAtomsSmallGlobalAll;
    uint32_t ligandAtomsSmallGlobalAllSize;
    uint32_t ligandAtomsSmallResultSize;

    DihedralRefDataGPU* dihedralRefData;
    uint32_t dihedralRefDataSize;

    CL_STRUCT_INT* equalsArray;
    uint32_t equalsArraySize;
    CL_STRUCT_INT* receptorIndex;
    uint32_t receptorIndexSize;
    CL_STRUCT_FLOAT* bestScore;
    uint32_t bestScoreSize;
    CL_STRUCT_FLOAT* score;
    uint32_t scoreSize;
    CL_STRUCT_INT* popNewIndex;
    uint32_t popNewIndexSize;
    cl_ulong* seed;
    uint32_t seedSize;
    LigandAtomPairsForClash* ligandAtomPairsForClash;
    uint32_t ligandAtomPairsForClashSize;
    tyche_i_state* rngStates;
    uint32_t rngStatesSize;

    uint32_t gridSize;

    // Timers:
    double t_dataPrep = 0.0;
    double tot_dataPrep = 0.0;
    double t_workerCreation = 0.0;
    double tot_workerCreation = 0.0;
    double t_dataToGPU = 0.0;
    double tot_dataToGPU = 0.0;
    double t_kernelCreation = 0.0;
    double tot_kernelCreation = 0.0;
    double t_kernelSetArgs = 0.0;
    double tot_kernelSetArgs = 0.0;
    double t_dataToCPU = 0.0;
    double tot_dataToCPU = 0.0;
    double t_kernelInit = 0.0;
    double tot_kernelInit = 0.0;
    double t_kernelInit2 = 0.0;
    double tot_kernelInit2 = 0.0;
    double t_kernelInit3 = 0.0;
    double tot_kernelInit3 = 0.0;
    double t_kernelInitGrid = 0.0;
    double tot_kernelInitGrid = 0.0;
    double t_kernelSyncToModel = 0.0;
    double tot_kernelSyncToModel = 0.0;
    double t_kernelPLP = 0.0;
    double tot_kernelPLP = 0.0;
    double t_kernelSort = 0.0;
    double tot_kernelSort = 0.0;
    double t_kernelNormalize = 0.0;
    double tot_kernelNormalize = 0.0;
    double t_kernelCreateNew = 0.0;
    double tot_kernelCreateNew = 0.0;
    double t_kernelFinalize = 0.0;
    double tot_kernelFinalize = 0.0;

    Data(std::string file, Batch& batchRef);
    ~Data();

    void trackScore();
};
