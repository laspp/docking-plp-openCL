#ifndef CLASSIFICATION_CL_H
#define CLASSIFICATION_CL_H

#include <clStructs.h>
#include <PLPConstants.cl>

inline int getPLPclass(int triposType) {
    return CLASSIFICATION_TABLE[triposType];
}

inline int getPLPinteraction(int class1, int class2) {
    return INTERACTION_TABLE[class1 * 7 + class2];
}

inline int getCLASHclass(int triposType) {
    return CLASH_CLASSIFICATION_TABLE[triposType];
}

// Returns squared distance!
inline float getRclash(int class1, int class2, int numBondsBetween) {

    float distance = ( (float)( (class1 == PLP_CLASH_CLASS_1 && class2 == PLP_CLASH_CLASS_1) ||

                       (class1 == PLP_CLASH_CLASS_2 && class2 == PLP_CLASH_CLASS_3 && numBondsBetween == 3) ||
                       (class1 == PLP_CLASH_CLASS_3 && class2 == PLP_CLASH_CLASS_2 && numBondsBetween == 3)
            ) * 2.5f) +
            ( (float)( (class1 == PLP_CLASH_CLASS_2 && class2 == PLP_CLASH_CLASS_3 && numBondsBetween > 3) ||
                       (class1 == PLP_CLASH_CLASS_3 && class2 == PLP_CLASH_CLASS_2 && numBondsBetween > 3) ||

                       (class1 == PLP_CLASH_CLASS_1 && class2 == PLP_CLASH_CLASS_2 && numBondsBetween == 3) ||
                       (class1 == PLP_CLASH_CLASS_1 && class2 == PLP_CLASH_CLASS_3 && numBondsBetween == 3) ||
                       (class1 == PLP_CLASH_CLASS_2 && class2 == PLP_CLASH_CLASS_1 && numBondsBetween == 3) ||
                       (class1 == PLP_CLASH_CLASS_2 && class2 == PLP_CLASH_CLASS_2 && numBondsBetween == 3) ||
                       (class1 == PLP_CLASH_CLASS_3 && class2 == PLP_CLASH_CLASS_1 && numBondsBetween == 3) ||
                       (class1 == PLP_CLASH_CLASS_3 && class2 == PLP_CLASH_CLASS_3 && numBondsBetween == 3)
            ) * 2.75f) +
            ( (float)( (class1 == PLP_CLASH_CLASS_1 && class2 == PLP_CLASH_CLASS_2 && numBondsBetween > 3) ||
                       (class1 == PLP_CLASH_CLASS_1 && class2 == PLP_CLASH_CLASS_3 && numBondsBetween > 3) ||
                       (class1 == PLP_CLASH_CLASS_2 && class2 == PLP_CLASH_CLASS_1 && numBondsBetween > 3) ||
                       (class1 == PLP_CLASH_CLASS_2 && class2 == PLP_CLASH_CLASS_2 && numBondsBetween > 3) ||
                       (class1 == PLP_CLASH_CLASS_3 && class2 == PLP_CLASH_CLASS_1 && numBondsBetween > 3) ||
                       (class1 == PLP_CLASH_CLASS_3 && class2 == PLP_CLASH_CLASS_3 && numBondsBetween > 3)
            ) * 3.0f);

    return distance * distance;
}


/*
From:
Clark, Matthew, Richard D. Cramer III, and Nicole Van Opdenbosch.
"Validation of the general purpose tripos 5.2 force field."
Journal of computational chemistry 10.8 (1989): 982-1012.
*/
void setTorsionalKS(global DihedralRefDataGPU* dihedralRefData, global AtomGPU* ligandAtoms, global BondGPU* ligandBonds) {
    
    int atom1ID = dihedralRefData->atom1ID;
    int atom2ID = dihedralRefData->atom2ID;
    int atom3ID = dihedralRefData->atom3ID;
    int atom4ID = dihedralRefData->atom4ID;

    int atom1Index = atom1ID - 1;
    int atom2Index = atom2ID - 1;
    int atom3Index = atom3ID - 1;
    int atom4Index = atom4ID - 1;


    if(ligandAtoms[atom1Index].id != atom1ID) {
        printf("[kernelInit] [setTorsionalKS] [Atom ID(%d) and index(%d) MISMATCH]\n", ligandAtoms[atom1Index].id, atom1Index);
    }
    if(ligandAtoms[atom2Index].id != atom2ID) {
        printf("[kernelInit] [setTorsionalKS] [Atom ID(%d) and index(%d) MISMATCH]\n", ligandAtoms[atom2Index].id, atom2Index);
    }
    if(ligandAtoms[atom3Index].id != atom3ID) {
        printf("[kernelInit] [setTorsionalKS] [Atom ID(%d) and index(%d) MISMATCH]\n", ligandAtoms[atom3Index].id, atom3Index);
    }
    if(ligandAtoms[atom4Index].id != atom4ID) {
        printf("[kernelInit] [setTorsionalKS] [Atom ID(%d) and index(%d) MISMATCH]\n", ligandAtoms[atom4Index].id, atom4Index);
    }
    
    int atom1TT = ligandAtoms[atom1Index].triposType;
    int atom2TT = ligandAtoms[atom2Index].triposType;
    int atom3TT = ligandAtoms[atom3Index].triposType;
    int atom4TT = ligandAtoms[atom4Index].triposType;

    int bondID = dihedralRefData->bondID;
    int bondIndex = bondID - 1;

    if(ligandBonds[bondIndex].id != bondID) {
        printf("[kernelInit] [setTorsionalKS] [Bond ID(%d) and index(%d) MISMATCH]\n", ligandBonds[bondIndex].id, bondIndex);
    }

    float bond = ligandBonds[bondIndex].partialBondOrder;// Partial bond order (1.0, 1.5, 2.0, 3.0 etc)

    // Reclassify back to Basic Types:
    if(atom1TT==TRIPOS_TYPE_C_1_H1) {
        atom1TT = TRIPOS_TYPE_C_1;
    }else if(atom1TT==TRIPOS_TYPE_C_2_H1 || atom1TT==TRIPOS_TYPE_C_2_H2) {
        atom1TT = TRIPOS_TYPE_C_2;
    }else if(atom1TT==TRIPOS_TYPE_C_3_H1 || atom1TT==TRIPOS_TYPE_C_3_H2 || atom1TT==TRIPOS_TYPE_C_3_H3) {
        atom1TT = TRIPOS_TYPE_C_3;
    }else if(atom1TT== TRIPOS_TYPE_C_ar_H1) {
        atom1TT = TRIPOS_TYPE_C_ar;
    }else if(atom1TT== TRIPOS_TYPE_H_P) {
        atom1TT = TRIPOS_TYPE_H;
    }

    if(atom2TT==TRIPOS_TYPE_C_1_H1) {
        atom2TT = TRIPOS_TYPE_C_1;
    }else if(atom2TT==TRIPOS_TYPE_C_2_H1 || atom2TT==TRIPOS_TYPE_C_2_H2) {
        atom2TT = TRIPOS_TYPE_C_2;
    }else if(atom2TT==TRIPOS_TYPE_C_3_H1 || atom2TT==TRIPOS_TYPE_C_3_H2 || atom2TT==TRIPOS_TYPE_C_3_H3) {
        atom2TT = TRIPOS_TYPE_C_3;
    }else if(atom2TT== TRIPOS_TYPE_C_ar_H1) {
        atom2TT = TRIPOS_TYPE_C_ar;
    }else if(atom2TT== TRIPOS_TYPE_H_P) {
        atom2TT = TRIPOS_TYPE_H;
    }

    if(atom3TT==TRIPOS_TYPE_C_1_H1) {
        atom3TT = TRIPOS_TYPE_C_1;
    }else if(atom3TT==TRIPOS_TYPE_C_2_H1 || atom3TT==TRIPOS_TYPE_C_2_H2) {
        atom3TT = TRIPOS_TYPE_C_2;
    }else if(atom3TT==TRIPOS_TYPE_C_3_H1 || atom3TT==TRIPOS_TYPE_C_3_H2 || atom3TT==TRIPOS_TYPE_C_3_H3) {
        atom3TT = TRIPOS_TYPE_C_3;
    }else if(atom3TT== TRIPOS_TYPE_C_ar_H1) {
        atom3TT = TRIPOS_TYPE_C_ar;
    }else if(atom3TT== TRIPOS_TYPE_H_P) {
        atom3TT = TRIPOS_TYPE_H;
    }

    if(atom4TT==TRIPOS_TYPE_C_1_H1) {
        atom4TT = TRIPOS_TYPE_C_1;
    }else if(atom4TT==TRIPOS_TYPE_C_2_H1 || atom4TT==TRIPOS_TYPE_C_2_H2) {
        atom4TT = TRIPOS_TYPE_C_2;
    }else if(atom4TT==TRIPOS_TYPE_C_3_H1 || atom4TT==TRIPOS_TYPE_C_3_H2 || atom4TT==TRIPOS_TYPE_C_3_H3) {
        atom4TT = TRIPOS_TYPE_C_3;
    }else if(atom4TT== TRIPOS_TYPE_C_ar_H1) {
        atom4TT = TRIPOS_TYPE_C_ar;
    }else if(atom4TT== TRIPOS_TYPE_H_P) {
        atom4TT = TRIPOS_TYPE_H;
    }

    // set default if no classification:
    float k = 0.2f;
    float s = 3.0f;

    // First all 4 atoms, then 3, then 2:
    if( (atom1TT==TRIPOS_TYPE_O_2 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
        (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_O_2 && bond==1.0) ) {
        k = 0.7f;
        s = -3.0f;
    }else if(atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0){
        k = 0.04f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.126f;
        s = 3.0f;
    }else if(atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0){
        k = 0.5f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.126f;
        s = -3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.273f;
        s = -3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.126f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.126f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.274f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ) {
        k = 0.274f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ) {
        k = 0.274f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ) {
        k = 0.274f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.126f;
        s = -3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.126f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.126f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.274f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_3 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ||
              (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.32f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_2 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.126f;
        s = -3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_C_3 && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.126f;
        s = 3.0f;
    }else if( (atom1TT==TRIPOS_TYPE_H && atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && atom4TT==TRIPOS_TYPE_H && bond==1.0) ) {
        k = 0.274f;
        s = 3.0f;
    }else if(atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_C_1 && bond==3.0) {
        k = 0.0f;
        s = 1.0f;
    }else if(atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_C_1 && bond==1.0) {
        k = 0.0f;
        s = 1.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_1 && bond==1.0) ) {
        k = 0.0f;
        s = 1.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_C_2 && bond==2.0) ||
              (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_1 && bond==2.0) ) {
        k = 0.0f;
        s = 1.0f;
    }else if(atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_2 && bond==2.0) {
        k = 12.5f;
        s = -2.0f;
    }else if(atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) {
        k = 1.424f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_1 && bond==1.0) ) {
        k = 0.0f;
        s = 1.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.12f;
        s = -3.0f;
    }else if(atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) {
        k = 0.2f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_C_1 && bond==1.0) ) {
        k = 0.0f;
        s = 1.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.12f;
        s = -3.0f;
    }else if(atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_C_ar && bond==1.5) {
        k = 2.0f;
        s = -2.0f;
    }else if(atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) {
        k = 0.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_C_1 && bond==1.0) ) {
        k = 0.0f;
        s = 1.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_1 && atom3TT==TRIPOS_TYPE_N_2 && bond==2.0) ||
              (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_C_1 && bond==2.0) ) {
        k = 0.0f;
        s = 1.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_N_2 && bond==2.0) ||
              (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_C_2 && bond==2.0) ) {
        k = 12.0f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 12.0f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.4f;
        s = -3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 1.6f;
        s = -2.0f;
    }else if(atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_N_2 && bond==2.0) {
        k = 1.6f;
        s = -2.0f;
    }else if(atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_N_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 0.12f;
        s = -3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_N_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_3 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.2f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_N_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_3 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 0.12f;
        s = -3.0f;
    }else if(atom2TT==TRIPOS_TYPE_N_3 && atom3TT==TRIPOS_TYPE_N_3 && bond==1.0) {
        k = 0.2f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_N_am && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_am && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 6.46f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_N_am && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_am && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.2f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_N_am && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_am && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_N_am && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_am && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) ) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_N_3 && atom3TT==TRIPOS_TYPE_N_am && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_am && atom3TT==TRIPOS_TYPE_N_3 && bond==1.0) ) {
        k = 0.12f;
        s = -3.0f;
    }else if(atom2TT==TRIPOS_TYPE_N_am && atom3TT==TRIPOS_TYPE_N_am && bond==1.0) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_N_ar && bond==1.5) ||
              (atom2TT==TRIPOS_TYPE_N_ar && atom3TT==TRIPOS_TYPE_C_ar && bond==1.5) ) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_N_pl3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_pl3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 12.0f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_N_pl3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_pl3 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.4f;
        s = -3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_N_pl3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_pl3 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_N_pl3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_N_pl3 && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) ) {
        k = 1.6f;
        s = -2.0f;
    }else if(atom2TT==TRIPOS_TYPE_N_pl3 && atom3TT==TRIPOS_TYPE_N_pl3 && bond==1.0) {
        k = 1.6f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_O_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_O_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 5.8f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_O_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_O_3 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 1.2f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_O_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_O_3 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 1.2f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_N_2 && atom3TT==TRIPOS_TYPE_O_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_O_3 && atom3TT==TRIPOS_TYPE_N_2 && bond==1.0) ) {
        k = 1.0f;
        s = 2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_N_3 && atom3TT==TRIPOS_TYPE_O_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_O_3 && atom3TT==TRIPOS_TYPE_N_3 && bond==1.0) ) {
        k = 0.2f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_P_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_P_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 1.0f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_P_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_P_3 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.4f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_P_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_P_3 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 1.0f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_O_3 && atom3TT==TRIPOS_TYPE_P_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_P_3 && atom3TT==TRIPOS_TYPE_O_3 && bond==1.0) ) {
        k = 0.4f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_S_2 && bond==2.0) ||
              (atom2TT==TRIPOS_TYPE_S_2 && atom3TT==TRIPOS_TYPE_C_2 && bond==2.0) ) {
        k = 1.0f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_S_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_S_2 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.4f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_S_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_S_2 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 1.0f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_N_3 && atom3TT==TRIPOS_TYPE_S_2 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_S_2 && atom3TT==TRIPOS_TYPE_N_3 && bond==1.0) ) {
        k = 0.4f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_2 && atom3TT==TRIPOS_TYPE_S_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_S_3 && atom3TT==TRIPOS_TYPE_C_2 && bond==1.0) ) {
        k = 1.0f;
        s = -2.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_3 && atom3TT==TRIPOS_TYPE_S_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_S_3 && atom3TT==TRIPOS_TYPE_C_3 && bond==1.0) ) {
        k = 0.4f;
        s = 3.0f;
    }else if( (atom2TT==TRIPOS_TYPE_C_ar && atom3TT==TRIPOS_TYPE_S_3 && bond==1.0) ||
              (atom2TT==TRIPOS_TYPE_S_3 && atom3TT==TRIPOS_TYPE_C_ar && bond==1.0) ) {
        k = 1.0f;
        s = 3.0f;
    }else if(atom2TT==TRIPOS_TYPE_S_3 && atom3TT==TRIPOS_TYPE_S_3 && bond==1.0) {
        k = 4.0f;
        s = 3.0f;
    }
    
    dihedralRefData->k = k;
    dihedralRefData->s = s;
}

#endif
