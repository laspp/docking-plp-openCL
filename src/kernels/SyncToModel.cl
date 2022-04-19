#ifndef SYNC_TO_MODEL_CL_H
#define SYNC_TO_MODEL_CL_H

#include <clStructs.h>
#include <vector3.cl>
#include <Quat.cl>
#include <dsyevh3.cl>
#include <constants.cl>

// TODO: setModelValueOccupancyElement(...)

// Returns dihedral formed between 3 vectors
float Dihedral(global AtomGPUsmall* m_atom1, global AtomGPUsmall* m_atom2, global AtomGPUsmall* m_atom3, global AtomGPUsmall* m_atom4) {
    // 4 Atoms Recieved DONE!

    // 4 Coordinates From Atoms DONE!

    // 3 vectors
    float v1[3];
    float v2[3];
    float v3[3];
        // v1
    v1[0] = m_atom1->x - m_atom2->x;
    v1[1] = m_atom1->y - m_atom2->y;
    v1[2] = m_atom1->z - m_atom2->z;
        // v2
    v2[0] = m_atom2->x - m_atom3->x;
    v2[1] = m_atom2->y - m_atom3->y;
    v2[2] = m_atom2->z - m_atom3->z;
        // v3
    v3[0] = m_atom3->x - m_atom4->x;
    v3[1] = m_atom3->y - m_atom4->y;
    v3[2] = m_atom3->z - m_atom4->z;

    // Calculate the cross products
    float A[3];
    float B[3];
    float C[3];
    crossProduct3((float*)v1, (float*)v2, (float*)A);
    crossProduct3((float*)v2, (float*)v3, (float*)B);
    crossProduct3((float*)v2, (float*)A, (float*)C);

    // Calculate the distances
    float rA = lengthNorm3((float*)A);
    float rB = lengthNorm3((float*)B);
    float rC = lengthNorm3((float*)C);

    // Calculate the sin and cos
    float cos_phi = dotProduct3((float*)A, (float*)B) / (rA * rB);
    float sin_phi = dotProduct3((float*)C, (float*)B) / (rC * rB);

    // Get phi and convert to degrees
    float phi = -atan2(sin_phi, cos_phi);
    return phi * 180.0f / M_PI;
}

global AtomGPUsmall* getAtomGPUsmallFromID(global AtomGPUsmall* ligandAtoms,int id) {
    int i = id - 1;
    if(ligandAtoms[i].id == id) {
        return &(ligandAtoms[i]);
    } else {
        printf("[SyncToModel] [getAtomGPUsmallFromID] [Atom ID(%d) and index(%d) MISMATCH]\n", ligandAtoms[i].id, id);
        return &(ligandAtoms[i]);
    }
}

void setModelValueDihedral(global AtomGPUsmall* ligandAtoms, global float* individual, global DihedralRefDataGPU* dihedralRefData, constant parametersForGPU* parameters) {
    // For Every Dihedral
    int i;
    global AtomGPUsmall* m_atom1;
    global AtomGPUsmall* m_atom2;
    global AtomGPUsmall* m_atom3;
    global AtomGPUsmall* m_atom4;
    float delta;
    for(i = 0; i < parameters->numDihedralElements; i++) {
        m_atom1 = getAtomGPUsmallFromID(ligandAtoms, dihedralRefData[i].atom1ID);
        m_atom2 = getAtomGPUsmallFromID(ligandAtoms, dihedralRefData[i].atom2ID);
        m_atom3 = getAtomGPUsmallFromID(ligandAtoms, dihedralRefData[i].atom3ID);
        m_atom4 = getAtomGPUsmallFromID(ligandAtoms, dihedralRefData[i].atom4ID);

        delta = individual[i] - Dihedral(m_atom1, m_atom2, m_atom3, m_atom4);

        // Only rotate if delta is non-zero
        if (fabs(delta) > 0.001f) {

            // Coords of atom 1 ( m_atom2 !!! NOT 1 !! NOT MISTAKE ! )
            float coord1[3];
            coord1[0] = m_atom2->x;
            coord1[1] = m_atom2->y;
            coord1[2] = m_atom2->z;

            // Vector along the bond between atom 1 and atom 2 (rotation axis)
            // ( m_atom3 !!! NOT 2 !! NOT MISTAKE ! )
            float coord2[3];
            coord2[0] = m_atom3->x;
            coord2[1] = m_atom3->y;
            coord2[2] = m_atom3->z;
            float bondVector[3];
            subtract2Vectors3((float*)coord2, (float*)coord1, (float*)bondVector);
            float toOrigin[3];
            negateVector3((float*)coord1, (float*)toOrigin);
            float quat_v[3];
            float quat_s;
            RbtQuat((float*)bondVector, delta * M_PI / 180.0f, &quat_s, (float*)quat_v);
            int j;
            global AtomGPUsmall* tempAtom;
            float tempCoord[3];
            float translated[3];
            float rotated[3];
            float translated2[3];
            for(j=0; j < dihedralRefData[i].numRotAtoms; j++) {
                tempAtom = getAtomGPUsmallFromID(ligandAtoms, dihedralRefData[i].rotAtomsIDs[j]);
                tempCoord[0] = tempAtom->x;
                tempCoord[1] = tempAtom->y;
                tempCoord[2] = tempAtom->z;
                // Translate(toOrigin); (add coordinates together)
                add2Vectors3((float*)tempCoord, (float*)toOrigin, (float*)translated);
                // RotateUsingQuat(quat);
                RotateUsingQuat(&quat_s, (float*)quat_v, (float*)translated, (float*)rotated);
                // Translate(coord1); (add coordinates together)
                add2Vectors3((float*)rotated, (float*)coord1, (float*)translated2);

                tempAtom->x = translated2[0];
                tempAtom->y = translated2[1];
                tempAtom->z = translated2[2];
            }
        }
    }
}

// Returns center of mass of atoms in the list
void GetCenterOfMass(float* com, global AtomGPUsmall* atoms, int numAtoms) {
    // Default constructor (initialise to zero)
    com[0] = 0.0f;
    com[1] = 0.0f;
    com[2] = 0.0f;
    
    // Accumulate sum of mass*coord
    float totalMass = 0.0f;
    float tempMass = 0.0f;
    for(int i = 0; i < numAtoms; i++) {
        tempMass = atoms[i].atomicMass;
        com[0] += (tempMass * atoms[i].x);
        com[1] += (tempMass * atoms[i].y);
        com[2] += (tempMass * atoms[i].z);

        totalMass += tempMass;// GetTotalAtomicMass
    }
    // Divide by total mass
    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
}

void GetPrincipalAxes(PrincipalAxesSyncGPU* principalAxes, global AtomGPUsmall* atoms, global float* individual, constant parametersForGPU* parameters, int numAtoms) {
    const int N = 3;

    // TODO: No check for atomList.empty() !

    // TODO: Special case for water !

    // Construct default principal axes:
    principalAxes->com[0] = 0.0f;
    principalAxes->com[1] = 0.0f;
    principalAxes->com[2] = 0.0f;

    principalAxes->axis1[0] = 1.0f;
    principalAxes->axis1[1] = 0.0f;
    principalAxes->axis1[2] = 0.0f;

    principalAxes->axis2[0] = 0.0f;
    principalAxes->axis2[1] = 1.0f;
    principalAxes->axis2[2] = 0.0f;

    principalAxes->axis3[0] = 0.0f;
    principalAxes->axis3[1] = 0.0f;
    principalAxes->axis3[2] = 1.0f;

    principalAxes->moment1 = 1.0f;
    principalAxes->moment2 = 1.0f;
    principalAxes->moment3 = 1.0f;

    // Store center of mass of CURRENT MODEL !!!
    GetCenterOfMass((float*)principalAxes->com, atoms, numAtoms);

    // Construct the moment of inertia tensor
    float inertiaTensor[3*3];// cuda fix (was: float inertiaTensor[N*N])
    // Set to zero
    zerosNxN((float*)inertiaTensor, N);
    int i;
    for(i=0; i < numAtoms; i++) {
        // Vector from center of mass to atom
        float r[3];
        float atomCoord[3];
        atomCoord[0] = atoms[i].x;
        atomCoord[1] = atoms[i].y;
        atomCoord[2] = atoms[i].z;
        subtract2Vectors3((float*)atomCoord, (float*)principalAxes->com, (float*)r);
        // Atomic mass
        float m = atoms[i].atomicMass;
        float rx2 = r[0] * r[0];
        float ry2 = r[1] * r[1];
        float rz2 = r[2] * r[2];
        // Diagonal elements (moments of inertia)
        float dIxx = m * (ry2 + rz2); //=r^2 - x^2
        float dIyy = m * (rx2 + rz2); //=r^2 - y^2
        float dIzz = m * (rx2 + ry2); //=r^2 - z^2
        inertiaTensor[0] += dIxx;// [0][0]
        inertiaTensor[4] += dIyy;// [1][1]
        inertiaTensor[8] += dIzz;// [2][2]
        // Off-diagonal elements (products of inertia) - symmetric matrix
        float dIxy = m * r[0] * r[1];
        float dIxz = m * r[0] * r[2];
        float dIyz = m * r[1] * r[2];
        inertiaTensor[1] -= dIxy;// [0][1]
        inertiaTensor[3] -= dIxy;// [1][0]
        inertiaTensor[2] -= dIxz;// [0][2]
        inertiaTensor[6] -= dIxz;// [2][0]
        inertiaTensor[5] -= dIyz;// [1][2]
        inertiaTensor[7] -= dIyz;// [2][1]
    }

    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(inertiaTensor);

    // Eigen::VectorXd eigenValues = eigenSolver.eigenvalues().real();

    // Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors().real();

    float eigenVectors[3*3];// cuda fix (was: float eigenVectors[N*N])
    float eigenValues[3];// cuda fix (was: float eigenValues[N])
    if(dsyevh3((float*)inertiaTensor, (float*)eigenVectors, (float*)eigenValues) != 0) {
        // Failure
        printf("[SyncToModel] [GetPrincipalAxes] [Failed to get eigenVectors and eigenValues]\n");
    } else {
       // Success

       // eigenvectors already normalized (length=1)

       // FIXME: assert that .imag() is Zero(N) ????? (comment from original code)

        // Load the principal axes and moments into the return parameter
        // We need to sort these so that axis1 is the first principal axis, axis2 the
        // second, axis3 the third. With only three elements to sort, this is probably
        // as good a way as any:
        unsigned int idx1 = 0;
        unsigned int idx2 = 1;
        unsigned int idx3 = 2;
        float swap;
        if(eigenValues[idx1] > eigenValues[idx2]) {
            swap = idx1;
            idx1 = idx2;
            idx2 = swap;
        }
        if(eigenValues[idx1] > eigenValues[idx3]) {
            swap = idx1;
            idx1 = idx3;
            idx3 = swap;
        }
        if(eigenValues[idx2] > eigenValues[idx3]) {
            swap = idx2;
            idx2 = idx3;
            idx3 = swap;
        }
        principalAxes->axis1[0] = eigenVectors[idx1];// (0, idx1)
        principalAxes->axis1[1] = eigenVectors[3+idx1];// (1, idx1)
        principalAxes->axis1[2] = eigenVectors[6+idx1];// (2, idx1)

        principalAxes->axis2[0] = eigenVectors[idx2];// (0, idx2)
        principalAxes->axis2[1] = eigenVectors[3+idx2];// (1, idx2)
        principalAxes->axis2[2] = eigenVectors[6+idx2];// (2, idx2)

        principalAxes->axis3[0] = eigenVectors[idx3];// (0, idx3)
        principalAxes->axis3[1] = eigenVectors[3+idx3];// (1, idx3)
        principalAxes->axis3[2] = eigenVectors[6+idx3];// (2, idx3)

        principalAxes->moment1 = eigenValues[idx1];
        principalAxes->moment2 = eigenValues[idx2];
        principalAxes->moment3 = eigenValues[idx3];

        // DM 28 Jun 2001 - for GA crossovers in particular we need to ensure the
        // principal axes returned are always consistent for a given ligand
        // conformation. Currently the direction of the axes vectors is not controlled
        // as for e.g. (+1,0,0) is degenerate with (-1,0,0). Method is to arbitrarily
        // check the first atom and invert the axes vectors if necessary to ensure the
        // atom coords lie in the positive quadrant of the coordinate space of
        // principal axes 1 and 2 Principal axis 3 is then defined to give a
        // right-handed set of axes
        //
        // LIMITATION: If atom 1 lies exactly on PA#1 or PA#2 this check will fail.
        // Ideally we would like to test an atom on the periphery of the molecule.
        float c0[3];
        float firstAtomCoords[3];
        firstAtomCoords[0] = atoms[0].x;
        firstAtomCoords[1] = atoms[0].y;
        firstAtomCoords[2] = atoms[0].z;
        subtract2Vectors3((float*)firstAtomCoords, (float*)(principalAxes->com), (float*)c0);
        float d1 = dotProduct3((float*)c0, (float*)(principalAxes->axis1));
        float d2 = dotProduct3((float*)c0, (float*)(principalAxes->axis2));
        if (d1 < 0.0f) {
            negateVector3((float*)(principalAxes->axis1), (float*)(principalAxes->axis1));
        }
        if (d2 < 0.0f) {
            negateVector3((float*)(principalAxes->axis2), (float*)(principalAxes->axis2));
        }
        crossProduct3((float*)(principalAxes->axis1), (float*)(principalAxes->axis2), (float*)(principalAxes->axis3));
        
    }
    
}

// For COM and Orientation, Only Once!
void setModelValuePosition(global AtomGPUsmall* ligandAtoms, global float* individual, constant parametersForGPU* parameters) {
    
    // Determine the principal axes and centre of mass of the reference atoms
    PrincipalAxesSyncGPU prAxes;
        // TODO: if tethered use tetheredAtomList, not whole atomList.
    // m_refAtoms = atomList (all atoms) = ligandAtoms
    GetPrincipalAxes(&prAxes, ligandAtoms, individual, parameters, parameters->ligandNumAtoms);

    // Determine the overall rotation required.
    // 1) Go back to realign with Cartesian axes
    PrincipalAxesSyncGPU CARTESIAN_AXES;
    CARTESIAN_AXES.com[0] = 0.0f;
    CARTESIAN_AXES.com[1] = 0.0f;
    CARTESIAN_AXES.com[2] = 0.0f;

    CARTESIAN_AXES.axis1[0] = 1.0f;
    CARTESIAN_AXES.axis1[1] = 0.0f;
    CARTESIAN_AXES.axis1[2] = 0.0f;

    CARTESIAN_AXES.axis2[0] = 0.0f;
    CARTESIAN_AXES.axis2[1] = 1.0f;
    CARTESIAN_AXES.axis2[2] = 0.0f;

    CARTESIAN_AXES.axis3[0] = 0.0f;
    CARTESIAN_AXES.axis3[1] = 0.0f;
    CARTESIAN_AXES.axis3[2] = 1.0f;

    CARTESIAN_AXES.moment1 = 1.0f;
    CARTESIAN_AXES.moment2 = 1.0f;
    CARTESIAN_AXES.moment3 = 1.0f;

    float qBack_v[3];
    float qBack_s;
    GetQuatFromAlignAxes(&prAxes, &CARTESIAN_AXES, &qBack_s, (float*)qBack_v);

    // 2) Go forward to the desired orientation
    float qForward_v[3];
    float qForward_s;
    float orientation[3];
    int startOrientationIndex = (parameters->chromStoreLen) - CHROM_SUBTRACT_FOR_ORIENTATION_1;
    orientation[0] = individual[startOrientationIndex];
    orientation[1] = individual[startOrientationIndex + 1];
    orientation[2] = individual[startOrientationIndex + 2];
    ToQuat((float*)orientation, &qForward_s, (float*)qForward_v);

    // 3 Combine the two rotations
    float q_v[3];
    float q_s;
    multiplyQuatResult(&qForward_s, (float*)qForward_v, &qBack_s, (float*)qBack_v, &q_s, (float*)q_v);// RbtQuat q = qForward * qBack;
    
    
    int i;
    global AtomGPUsmall* tempAtom;
    float tempCoord[3];
    float negPrAxesCom[3];
    negateVector3((float*)prAxes.com, (float*)negPrAxesCom);
    float translated[3];
    float rotated[3];
    float com[3];
    int startComIndex = (parameters->chromStoreLen) - CHROM_SUBTRACT_FOR_CENTER_OF_MASS_1;
    com[0] = individual[startComIndex];
    com[1] = individual[startComIndex + 1];
    com[2] = individual[startComIndex + 2];
    float translated2[3];
    for(i = 0; i < parameters->ligandNumAtoms; i++) {
        tempAtom = &(ligandAtoms[i]);

        tempCoord[0] = tempAtom->x;
        tempCoord[1] = tempAtom->y;
        tempCoord[2] = tempAtom->z;

        // Move to origin
        add2Vectors3((float*)tempCoord, (float*)negPrAxesCom, (float*)translated);
        // Rotate
        RotateUsingQuat(&q_s, (float*)q_v, (float*)translated, (float*)rotated);
        // Move to new centre of mass
        add2Vectors3((float*)rotated, (float*)com, (float*)translated2);

        tempAtom->x = translated2[0];
        tempAtom->y = translated2[1];
        tempAtom->z = translated2[2];

    }
    
}

void syncToModel(global AtomGPUsmall* ligandAtoms, global float* individual, global DihedralRefDataGPU* dihedralRefData, constant parametersForGPU* parameters) {
    // TODO: setModelValueOccupancyElement(...)
    setModelValueDihedral(ligandAtoms, individual, dihedralRefData, parameters);
    setModelValuePosition(ligandAtoms, individual, parameters);
}

#endif
