#ifndef CL_STRUCT_H
#define CL_STRUCT_H

#ifdef CL_STRUCT_HOST
    #if defined(__APPLE__) || defined(__MACOSX)
        #include <OpenCL/opencl.h>
    #else
        #include <CL/opencl.h>
    #endif
    #define CL_STRUCT_INT cl_int
    #define CL_STRUCT_FLOAT cl_float
#else
    #define CL_STRUCT_INT int
    #define CL_STRUCT_FLOAT float
#endif

#define TOTAL_PLP_CLASSES 6

typedef struct CoordGPU_t {
    CL_STRUCT_FLOAT x;
    CL_STRUCT_FLOAT y;
    CL_STRUCT_FLOAT z;
} CoordGPU;

typedef struct PrincipalAxesGPU_t {
    CoordGPU center;
    CoordGPU axis1;
    CoordGPU axis2;
    CoordGPU axis3;
    CL_STRUCT_FLOAT moment1;
    CL_STRUCT_FLOAT moment2;
    CL_STRUCT_FLOAT moment3;
} PrincipalAxesGPU;

typedef struct CavityInfoGPU_t {
    PrincipalAxesGPU principalAxes;
    CL_STRUCT_FLOAT volume;
    CL_STRUCT_INT numCoords;
    CoordGPU gridStep;
    CoordGPU minCoord;
    CoordGPU maxCoord;
} CavityInfoGPU;

typedef struct DockingSiteInfoGPU_t {
    // box Receptor Atoms:
    CoordGPU minReceptor;
    CoordGPU maxReceptor;
    // box Cavity:
    CoordGPU minCavity;
    CoordGPU maxCavity;
} DockingSiteInfo;

typedef struct GridGPU_t {
    CL_STRUCT_FLOAT gridStep[3];// Grid step in X,Y,Z
    CL_STRUCT_FLOAT gridMin[3];// Min Coord
    CL_STRUCT_FLOAT gridMax[3];// Max Coord
    CL_STRUCT_INT NXYZ[3];// Size of X,Y,Z dim
    CL_STRUCT_INT N;// Total number of grid points
    CL_STRUCT_INT totalN;// Total number of grid points for all plp classes
    CL_STRUCT_INT SXYZ[3];// Stride of X (NY * NZ), Y (NZ), Z (1)
    CL_STRUCT_INT totalPLPClasses;
} GridGPU;

typedef struct GridPointGPU_t {
    CL_STRUCT_FLOAT point[TOTAL_PLP_CLASSES];
} GridPointGPU;

typedef struct parametersForGPU_t {

    // Basic Parameters
    CL_STRUCT_INT nruns; // Number of Populations
    CL_STRUCT_INT popSize; // Number of individuals in each population ! IF ERROR INCREASE CPU AND GPU STRUCT !
    CL_STRUCT_INT popMaxSize; // Needed for Buffer Allocation (imported not good =popSize)
    CL_STRUCT_INT maxThreads; // Max Number of Threads
    CL_STRUCT_INT maxDoubleLength; // Max Length of Double Array
    CL_STRUCT_INT chromStoreLen; // Number of floats for one individual
    // Number of Parameters
    CL_STRUCT_INT ligandNumAtoms;
    CL_STRUCT_INT ligandNumBonds;
    CL_STRUCT_INT receptorNumAtoms;
    CL_STRUCT_INT receptorNumBonds;
    // Ligand Parameters
    CL_STRUCT_FLOAT dihedral_step;
    CL_STRUCT_FLOAT rot_step;
    CL_STRUCT_FLOAT trans_step;
    CL_STRUCT_FLOAT max_dihedral;
    CL_STRUCT_FLOAT max_rot;
    CL_STRUCT_FLOAT max_trans;
    CL_STRUCT_INT transMode;
    CL_STRUCT_INT rotMode;
    CL_STRUCT_INT dihedralMode;
    // GA Parameters TODO: Multiple different GA Runs
    CL_STRUCT_INT history_freq;
    CL_STRUCT_INT nconvergence;
    CL_STRUCT_INT ncycles;
    CL_STRUCT_FLOAT  equality_threshold;
    CL_STRUCT_FLOAT  new_fraction;
    CL_STRUCT_INT nReplicates;
    CL_STRUCT_INT nReplicatesNumThreads;
    CL_STRUCT_FLOAT pcrossover;
    CL_STRUCT_FLOAT step_size;
    CL_STRUCT_INT cmutate;
    CL_STRUCT_INT  xovermut;
    // Cavity Info
    CavityInfoGPU cavityInfo;
    // Dihedral
    CL_STRUCT_INT numDihedralElements;
    // Docking site info
    DockingSiteInfo dockingSiteInfo;
    // numLigandAtomPairsForClash
    CL_STRUCT_INT numLigandAtomPairsForClash;
    CL_STRUCT_INT numUniqueAtom1PairsForClash;
    // Grid
    GridGPU grid;
} parametersForGPU;

typedef struct AtomGPU_t {
    CL_STRUCT_INT id;
    CL_STRUCT_INT atomicNo;
    CL_STRUCT_INT numImplicitHydrogens;
    CL_STRUCT_INT triposType;
    CL_STRUCT_INT classification;// classified atom
    CL_STRUCT_FLOAT atomicMass;
    CL_STRUCT_FLOAT formalCharge;
    CL_STRUCT_FLOAT partialCharge;
    CL_STRUCT_FLOAT groupCharge;
    CL_STRUCT_FLOAT x;
    CL_STRUCT_FLOAT y;
    CL_STRUCT_FLOAT z;
    CL_STRUCT_FLOAT vdwRadius;
    CL_STRUCT_INT cyclicFlag;
    CL_STRUCT_INT enabled;
    CL_STRUCT_INT good;
} AtomGPU;

typedef struct AtomGPUsmall_t {
    CL_STRUCT_INT id;
    CL_STRUCT_INT atomicNo;
    CL_STRUCT_INT triposType;
    CL_STRUCT_INT classification;// classified atom
    CL_STRUCT_FLOAT atomicMass;
    CL_STRUCT_FLOAT x;
    CL_STRUCT_FLOAT y;
    CL_STRUCT_FLOAT z;
} AtomGPUsmall;

typedef struct BondGPU_t {
    CL_STRUCT_INT id;
    CL_STRUCT_INT atom1ID;
    CL_STRUCT_INT atom2ID;
    CL_STRUCT_INT formalBondOrder;// bond type (single, CL_STRUCT_FLOAT, tripple bond)
    CL_STRUCT_FLOAT partialBondOrder;
    CL_STRUCT_INT cyclicFlag;
    CL_STRUCT_INT isRotatable;
} BondGPU;

typedef struct DihedralRefDataGPU_t {
    CL_STRUCT_INT bondID;        // ID of a bond that corresponds to the dihedral element
    CL_STRUCT_INT atom1ID;
    CL_STRUCT_INT atom2ID;
    CL_STRUCT_INT atom3ID;
    CL_STRUCT_INT atom4ID;
    CL_STRUCT_INT numRotAtoms;   // number of rotatable atoms
    CL_STRUCT_FLOAT stepSize;        // dihedral step size - same as dihedral_step in LigandFlexData

    CL_STRUCT_INT rotAtomsIDs[200];  // array of rotatable atoms IDs, ! IF ERROR INCREASE CPU AND GPU STRUCT !

    CL_STRUCT_FLOAT k;
    CL_STRUCT_FLOAT s;
} DihedralRefDataGPU;

// For Sync to Model
typedef struct PrincipalAxesSyncGPU_t {
    CL_STRUCT_FLOAT com[3];
    CL_STRUCT_FLOAT axis1[3];
    CL_STRUCT_FLOAT axis2[3];
    CL_STRUCT_FLOAT axis3[3];
    CL_STRUCT_FLOAT moment1;
    CL_STRUCT_FLOAT moment2;
    CL_STRUCT_FLOAT moment3;
} PrincipalAxesSyncGPU;

typedef struct LigandAtomPairsForClash_t {
    CL_STRUCT_INT numBondsBetween;
    CL_STRUCT_INT atom1ID;
    CL_STRUCT_INT atom2ID;
    CL_STRUCT_FLOAT rClash;
    CL_STRUCT_INT numAtom1;
} LigandAtomPairsForClash;

#endif
