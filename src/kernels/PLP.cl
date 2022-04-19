/*
From:
 J. Chem. Inf. Model. 2009, 49, 84–96
Empirical Scoring Functions for Advanced Protein-Ligand Docking with PLANTS
Oliver Korb,† Thomas Stützle,‡ and Thomas E. Exner*,†
Theoretische Chemische Dynamik, Fachbereich Chemie, Universität Konstanz, 78457 Konstanz, Germany,
and IRIDIA, CoDE, Université Libre de Bruxelles, Brussels, Belgium

Using PLANTS Model5
*/

#ifndef PLP_CL_H
#define PLP_CL_H

#include <clStructs.h>

#include <PLPConstants.cl>
#include <classification.cl>
#include <vector3.cl>

inline float plp(float r, float A, float B, float C, float D, float E, float F) {
    
    return (r < A) * (F * (A - r) / A)
            + (A <= r && r < B) * (E * (r - A) / (B - A))
            + (B <= r && r < C) * (E)
            + (C <= r && r <= D) * (E * (D - r) / (D - C));
}

inline float rep(float r, float A, float B, float C, float D) {

    return (r < A) * (r * (C - D) / A + D)
            + (A <= r && r <= B) * (-C * (r - A) / (B - A) + C);
}

float FplpActual(float* v1, int classification, global AtomGPU* rAtom) {

    int interaction = getPLPinteraction(classification, rAtom->classification);
    // receptor atom coordinates:
    float v2[3];
    v2[0] = rAtom->x;
    v2[1] = rAtom->y;
    v2[2] = rAtom->z;
    // Distance
    float r = distance2Points3((float*)v1, (float*)v2);

    return (interaction == PLP_INTERACTION_hbond ||
            interaction == PLP_INTERACTION_metal ||
            interaction == PLP_INTERACTION_buried ||
            interaction == PLP_INTERACTION_nonpolar)
                *
            plp(r, FPLP_TABLE[interaction * 6], FPLP_TABLE[interaction * 6 + 1],
                   FPLP_TABLE[interaction * 6 + 2], FPLP_TABLE[interaction * 6 + 3],
                   FPLP_TABLE[interaction * 6 + 4], 20.0f)
            + 
            (interaction == PLP_INTERACTION_repulsive)
                *
            rep(r, 3.2f, 5.0f, W_PLP_REP_M5 * 0.1f, W_PLP_REP_M5 * 20.0f);
}

float Fplp(global AtomGPUsmall* lAtom, global AtomGPU* rAtom) {

    // ligand atom coordinates:
    float v1[3];
    v1[0] = lAtom->x;
    v1[1] = lAtom->y;
    v1[2] = lAtom->z;

    return FplpActual((float*)v1, lAtom->classification, rAtom);
}

float CsiteAtom(global AtomGPUsmall* lAtom, constant parametersForGPU* parameters) {

    float atom[3];
    atom[0] = lAtom->x;
    atom[1] = lAtom->y;
    atom[2] = lAtom->z;
    float min[3];
    min[0] = parameters->dockingSiteInfo.minCavity.x;
    min[1] = parameters->dockingSiteInfo.minCavity.y;
    min[2] = parameters->dockingSiteInfo.minCavity.z;
    float max[3];
    max[0] = parameters->dockingSiteInfo.maxCavity.x;
    max[1] = parameters->dockingSiteInfo.maxCavity.y;
    max[2] = parameters->dockingSiteInfo.maxCavity.z;

    // if heavy atom outside, constant penalty
    return (lAtom->triposType != TRIPOS_TYPE_H && lAtom->triposType != TRIPOS_TYPE_H_P && 
            (atom[0] < min[0] || atom[0] > max[0] ||
             atom[1] < min[1] || atom[1] > max[1] ||
             atom[2] < min[2] || atom[2] > max[2])) * 50.0f;
}

inline int CsiteReceptor(float x, float y, float z, constant parametersForGPU* parameters) {
    float min[3];
    min[0] = parameters->dockingSiteInfo.minReceptor.x;
    min[1] = parameters->dockingSiteInfo.minReceptor.y;
    min[2] = parameters->dockingSiteInfo.minReceptor.z;
    float max[3];
    max[0] = parameters->dockingSiteInfo.maxReceptor.x;
    max[1] = parameters->dockingSiteInfo.maxReceptor.y;
    max[2] = parameters->dockingSiteInfo.maxReceptor.z;
    
    return (x > min[0] && x < max[0] && y > min[1] && y < max[1] && z > min[2] && z < max[2]);
}

float PLPclash(global AtomGPUsmall* atom1, global AtomGPUsmall* atom2, float rClash) {

    float atom1COORD[3];
    atom1COORD[0] = atom1->x;
    atom1COORD[1] = atom1->y;
    atom1COORD[2] = atom1->z;

    float atom2COORD[3];
    atom2COORD[0] = atom2->x;
    atom2COORD[1] = atom2->y;
    atom2COORD[2] = atom2->z;

    float r = distance2Points3((float*)atom1COORD, (float*)atom2COORD);
    
    return ( (float)( r < rClash) * ( ( (rClash*rClash - r*r) / (rClash*rClash) ) * 50.0f ));
}

/*
From:
Clark, Matthew, Richard D. Cramer III, and Nicole Van Opdenbosch.
"Validation of the general purpose tripos 5.2 force field."
Journal of computational chemistry 10.8 (1989): 982-1012.
*/
inline float PLPtorsional(float angle, float k, float s) {

    return k * (1.0f + s/fabs(s) * cos(fabs(s) * angle));
}

#endif
