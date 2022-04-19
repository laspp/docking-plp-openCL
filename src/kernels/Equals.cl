#ifndef EQUALS_CL_H
#define EQUALS_CL_H

#include <clStructs.h>
#include <constants.cl>

#include <mutate.cl>
#include <vector3.cl>

// TODO: CompareVectorOccupancyElement(...)

float CompareVectorDihedral(global float* g1, global float* g2, constant parametersForGPU* parameters) {

    float retVal = 0.0f;
    float absDiff;
    
    float stepSize = parameters->dihedral_step;
    if (stepSize > 0.0f) {
        for(int i = 0; i < parameters->numDihedralElements; i++) {
            absDiff = fabs(StandardisedValue(g1[i] - g2[i]));
            retVal = fmax(retVal, absDiff / stepSize);
        }
    }
    return retVal;
}

float CompareVectorPosition(global float* g1, global float* g2, constant parametersForGPU* parameters) {
    
    float retVal = 0.0f;

    // COM:
    if(parameters->transMode != CHROM_m_mode_eMode_FIXED) {

        int startComIndex = (parameters->chromStoreLen) - CHROM_SUBTRACT_FOR_CENTER_OF_MASS_1;
        float com1[3];
        com1[0] = g1[startComIndex];
        com1[1] = g1[startComIndex + 1];
        com1[2] = g1[startComIndex + 2];
        float com2[3];
        com2[0] = g2[startComIndex];
        com2[1] = g2[startComIndex + 1];
        com2[2] = g2[startComIndex + 2];

        float transStepSize = parameters->trans_step;
        // Compare distance between centres of mass
        if (transStepSize > 0.0f) {
            float absDiff = distance2Points3((float*)com1, (float*)com2);
            float relDiff = absDiff / transStepSize;
            retVal = fmax(retVal, relDiff);
        }
    }
    // Orientation:
    if(parameters->rotMode != CHROM_m_mode_eMode_FIXED) {

        int startOrientationIndex = (parameters->chromStoreLen) - CHROM_SUBTRACT_FOR_ORIENTATION_1;
        float orientation1[3];
        orientation1[0] = g1[startOrientationIndex];
        orientation1[1] = g1[startOrientationIndex + 1];
        orientation1[2] = g1[startOrientationIndex + 2];
        float orientation2[3];
        orientation2[0] = g2[startOrientationIndex];
        orientation2[1] = g2[startOrientationIndex + 1];
        orientation2[2] = g2[startOrientationIndex + 2];

        float rotStepSize = parameters->rot_step;
        // Compare orientations
        if (rotStepSize > 0.0f) {
            // Determine the difference between the two orientations
            // in terms of the axis/angle needed to align them
            // q.s = std::cos(phi / 2)
            float otherOrientation_v[3];
            float otherOrientation_s;
            ToQuat((float*)orientation2, &otherOrientation_s, (float*)otherOrientation_v);

            float m_orientation_v[3];
            float m_orientation_s;
            ToQuat((float*)orientation1, &m_orientation_s, (float*)m_orientation_v);

            float m_orientation_Conj_v[3];
            float m_orientation_Conj_s;
            ConjQuat(&m_orientation_s, (float*)m_orientation_v, &m_orientation_Conj_s, (float*)m_orientation_Conj_v);

            float qAlign_v[3];
            float qAlign_s;
            multiplyQuatResult(&otherOrientation_s, (float*)otherOrientation_v, &m_orientation_Conj_s, (float*)m_orientation_Conj_v, &qAlign_s, (float*)qAlign_v);

            float cosHalfTheta = qAlign_s;
            if (cosHalfTheta < -1.0f) {
                cosHalfTheta = -1.0f;
            } else if (cosHalfTheta > 1.0f) {
                cosHalfTheta = 1.0f;
            }
            float absDiff = fabs(StandardisedValueRotation(2.0f * acos(cosHalfTheta)));
            float relDiff = absDiff / rotStepSize;
            retVal = fmax(retVal, relDiff);
        }
    }
    return retVal;
}

// 1 = not equals, 0 = equals
int EqualsGenome(global float* g1, global float* g2, constant parametersForGPU* parameters) {
    
    // No check for same length!

    // Checks for maximum difference of any of the underlying chromosome elements;
    // Dihedral element has one float
    // COM and Orientation has 3 each.

    // TODO: CompareVectorOccupancyElement(...)
    float retValDihedral = CompareVectorDihedral(g1, g2, parameters);
    float retValPosition = CompareVectorPosition(g1, g2, parameters);
    float retVal = fmax(retValDihedral, retValPosition);

    return (retVal > parameters->equality_threshold);
}

#endif
