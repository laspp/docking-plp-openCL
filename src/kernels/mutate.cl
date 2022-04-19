#ifndef MUTATE_CL_H
#define MUTATE_CL_H

#include <clStructs.h>
#include <constants.cl>
#include <PLPConstants.cl>

#include <tyche_i.cl>
#include <randomExtra.cl>
#include <vector3.cl>
#include <Quat.cl>

// MUTATE DIHEDRAL:

float StandardisedValue(float dihedralAngle) {
  while (dihedralAngle >= 180.0f) {
    dihedralAngle -= 360.0f;
  }
  while (dihedralAngle < -180.0f) {
    dihedralAngle += 360.0f;
  }
  return dihedralAngle;
}

float CorrectTetheredDihedral(float initialValue, float m_value, constant parametersForGPU* parameters) {
  float maxDelta = parameters->max_dihedral;

  float delta = StandardisedValue(m_value - initialValue);
  if (delta > maxDelta) {
    return StandardisedValue(initialValue + maxDelta);
  } else if (delta < -maxDelta) {
    return StandardisedValue(initialValue - maxDelta);
  } else {
      return m_value;//DO NOTHING (aka. return current value)
  }
}

void mutateDihedral(global float* chromosomeDihedral, int numOfRotatableBonds, float relStepSize, tyche_i_state* state, constant parametersForGPU* parameters) {
    
    float absStepSize = relStepSize * parameters->dihedral_step;
    float delta;
    int m_mode;
    float initialValue;

    float tempDihedral;

    for(int i = 0; i < numOfRotatableBonds; i++){
 
        tempDihedral = chromosomeDihedral[i];

        if (absStepSize > 0.0f) {

            m_mode = parameters->dihedralMode;
            
            if(m_mode == CHROM_m_mode_eMode_TETHERED) {

                delta = 2.0f * absStepSize * tyche_i_float(state[0]) - absStepSize;
                initialValue = tempDihedral;
                tempDihedral = StandardisedValue(tempDihedral + delta);
                tempDihedral = CorrectTetheredDihedral(initialValue, tempDihedral, parameters);

            } else if(m_mode == CHROM_m_mode_eMode_FREE) {

                delta = 2.0f * absStepSize * tyche_i_float(state[0]) - absStepSize;
                tempDihedral = StandardisedValue(tempDihedral + delta);

            }
        }

        chromosomeDihedral[i] = tempDihedral;
    }
}

// MUTATE CENTER OF MASS:

void CorrectTetheredCOM(float* chromosomeCOM, float* initCOM, constant parametersForGPU* parameters) {

    float axis[3];
    float unit[3];
    float maxTrans = parameters->max_trans;

    // vector from initial COM to new point
    axis[0] = chromosomeCOM[0] - initCOM[0];
    axis[1] = chromosomeCOM[1] - initCOM[1];
    axis[2] = chromosomeCOM[2] - initCOM[2];

    // avoid square roots until necessary
    float length = lengthNorm3((float*)axis);
    float length2 = length * length;
    unitVector3((float*)axis, (float*)unit, length);
    if (length2 > (maxTrans * maxTrans)) {
        // Stay just inside the boundary sphere
       chromosomeCOM[0] = initCOM[0] + (0.999f * maxTrans * unit[0]);
       chromosomeCOM[1] = initCOM[1] + (0.999f * maxTrans * unit[1]);
       chromosomeCOM[2] = initCOM[2] + (0.999f * maxTrans * unit[2]);
    }
}

void mutateCOM(global float* chromosomeCOM, float relStepSize, tyche_i_state* state, constant parametersForGPU* parameters) {
    //chromosomeCOM[0] chromosomeCOM[1] chromosomeCOM[2]
    float absTransStepSize;
    float dist;
    float axis[3];
    float initCOM[3];

    int m_transMode = parameters->transMode;

    float tempCOM[3];
    tempCOM[0] = chromosomeCOM[0];
    tempCOM[1] = chromosomeCOM[1];
    tempCOM[2] = chromosomeCOM[2];
    
    if(m_transMode == CHROM_m_mode_eMode_TETHERED) {
        absTransStepSize = relStepSize * parameters->trans_step;
        if (absTransStepSize > 0.0f) {
            dist = absTransStepSize * tyche_i_float(state[0]);
            //Get random unit vector to axis
            randomUnitVector3((float*)axis, state);            
            //save init COM
            saveVector3((float*)tempCOM, (float*)initCOM);

            tempCOM[0] += dist * axis[0];
            tempCOM[1] += dist * axis[1];
            tempCOM[2] += dist * axis[2];

            CorrectTetheredCOM((float*)tempCOM, (float*)initCOM, parameters);
        }
    } else if(m_transMode == CHROM_m_mode_eMode_FREE) {
        absTransStepSize = relStepSize * parameters->trans_step;
        if (absTransStepSize > 0.0f) {
            dist = absTransStepSize * tyche_i_float(state[0]);
            //Get random unit vector to axis
            randomUnitVector3((float*)axis, state);

            tempCOM[0] += dist * axis[0];
            tempCOM[1] += dist * axis[1];
            tempCOM[2] += dist * axis[2];

        }
    }
    // FIXED: Do nothing

    chromosomeCOM[0] = tempCOM[0];
    chromosomeCOM[1] = tempCOM[1];
    chromosomeCOM[2] = tempCOM[2];
}

// MUTATE ORIENTATION:

float StandardisedValueRotation(float rotationAngle) {
  while (rotationAngle >= M_PI) {
    rotationAngle -= 2.0f * M_PI;
  }
  while (rotationAngle < -M_PI) {
    rotationAngle += 2.0f * M_PI;
  }
  return rotationAngle;
}

void CorrectTetheredOrientation(float* chromosomeOrientation, float* initOrientation, constant parametersForGPU* parameters) {
    // Check for orientation out of bounds
    float maxRot = parameters->max_rot;
    
    float s_qAlign;
    float v_qAlign[3];

    float s_initial;
    float v_initial[3];

    float s_conj;
    float v_conj[3];

    ToQuat((float*)chromosomeOrientation, &s_initial, (float*)v_initial);

    ConjQuat(&s_initial, (float*)v_initial, &s_conj, (float*)v_conj);

    //Copy to
    s_qAlign = s_conj;
    saveVector3((float*)v_conj, (float*)v_qAlign);

    multiplyQuat(&s_initial, (float*)v_initial, &s_qAlign, (float*)v_qAlign);
    float cosHalfTheta = s_qAlign;
    if (cosHalfTheta < -1.0f) {
        cosHalfTheta = -1.0f;
    } else if (cosHalfTheta > 1.0f) {
        cosHalfTheta = 1.0f;
    }
    // Theta is the rotation angle required to realign with reference
    float theta = StandardisedValueRotation(2.0f * acos(cosHalfTheta));
    // Deal with pos and neg angles independently as we have to
    // invert the rotation axis for negative angles
    float axis[3];
    float negate[3];
    if (theta < -maxRot) {

        negateVector3((float*)v_qAlign, (float*)negate);

        divideVectorByScalar3((float*)negate, sin(theta / 2.0f), (float*)axis);

        // Adjust theta to bring the orientation just inside the tethered bound
        theta += 0.999f * maxRot;

        Rotate((float*)chromosomeOrientation, (float*)axis, theta);

    } else if (theta > maxRot) {
        
        divideVectorByScalar3((float*)v_qAlign, sin(theta / 2.0f), (float*)axis);

        // Adjust theta to bring the orientation just inside the tethered bound
        theta -= 0.999f * maxRot;

        Rotate((float*)chromosomeOrientation, (float*)axis, theta);
    }
}

void mutateOrientation(global float* chromosomeOrientation, float relStepSize, tyche_i_state* state, constant parametersForGPU* parameters) {
    //chromosomeOrientation[0] chromosomeOrientation[1] chromosomeOrientation[2]
    float absRotStepSize;
    float theta;

    float axis[3];
    float initOrientation[3];

    int m_rotMode = parameters->rotMode;

    float tempOrientation[3];
    tempOrientation[0] = chromosomeOrientation[0];
    tempOrientation[1] = chromosomeOrientation[1];
    tempOrientation[2] = chromosomeOrientation[2];
    
    if(m_rotMode == CHROM_m_mode_eMode_TETHERED) {
        absRotStepSize = relStepSize * parameters->rot_step;
        if (absRotStepSize > 0.0f) {
            theta = absRotStepSize * tyche_i_float(state[0]);

            // Get random unit vector to axis
            randomUnitVector3((float*)axis, state);
            // save init Orientation
            saveVector3((float*)tempOrientation, (float*)initOrientation);
            // Rotate:
            Rotate((float*)tempOrientation, (float*)axis, theta);
            
            CorrectTetheredOrientation((float*)tempOrientation, (float*)initOrientation, parameters);

        }
    } else if(m_rotMode == CHROM_m_mode_eMode_FREE) {
        absRotStepSize = relStepSize * parameters->rot_step;
        if (absRotStepSize > 0.0f) {
            theta = absRotStepSize * tyche_i_float(state[0]);

            //Get random unit vector to axis
            randomUnitVector3((float*)axis, state);
            // Rotate:
            Rotate((float*)tempOrientation, (float*)axis, theta);
        }
    }
    // FIXED: Do nothing

    chromosomeOrientation[0] = tempOrientation[0];
    chromosomeOrientation[1] = tempOrientation[1];
    chromosomeOrientation[2] = tempOrientation[2];
}

// MUTATE OCCUPANCY ELEMENT:

// mutateOccupancyElement(...) TODO


// MAIN MUTATE:

void mutate(global float* chromosome, int chromStoreLen, float relStepSize, tyche_i_state* state, constant parametersForGPU* parameters) {
    mutateDihedral(chromosome, chromStoreLen - CHROM_SUBTRACT_FOR_ROTATABLE_BONDS_LENGTH, relStepSize, state, parameters);
    mutateCOM(&(chromosome[chromStoreLen - CHROM_SUBTRACT_FOR_CENTER_OF_MASS_1]), relStepSize, state, parameters);
    mutateOrientation(&(chromosome[chromStoreLen - CHROM_SUBTRACT_FOR_ORIENTATION_1]), relStepSize, state, parameters);
    // mutateOccupancyElement(...) TODO
}

// CAUCHY MUTATE:

void CauchyMutate(global float* chromosome, int chromStoreLen, float mean, float variance, tyche_i_state* state, constant parametersForGPU* parameters) {
  // Need to convert the Cauchy random variable to a positive number
  // and use this as the relative step size for mutation
  
  float relStepSize = PositiveCauchyRandomWMeanVariance(mean, variance, state);
  
  mutate(chromosome, chromStoreLen, relStepSize, state, parameters);
}

#endif
