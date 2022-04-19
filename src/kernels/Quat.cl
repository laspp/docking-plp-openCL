#ifndef QUAT_CL_H
#define QUAT_CL_H

#include <clStructs.h>
#include <vector3.cl>

void RbtQuat(float* axis, float phi, float* s, float* v) {
    float halfPhi = 0.5f * phi;
    *s = cos(halfPhi);// Scalar component

    float unit[3];
    unitVector3((float*)axis, (float*)unit, lengthNorm3((float*)axis));

    // Vector component
    v[0] = sin(halfPhi) * unit[0];
    v[1] = sin(halfPhi) * unit[1];
    v[2] = sin(halfPhi) * unit[2];
}

void ToQuat(float* chromosomeOrientation, float* s, float* v) {
  float c1 = cos(chromosomeOrientation[0] / 2.0f);
  float s1 = sin(chromosomeOrientation[0] / 2.0f);
  float c2 = cos(chromosomeOrientation[1] / 2.0f);
  float s2 = sin(chromosomeOrientation[1] / 2.0f);
  float c3 = cos(chromosomeOrientation[2] / 2.0f);
  float s3 = sin(chromosomeOrientation[2] / 2.0f);
  float c1c2 = c1 * c2;
  float s1s2 = s1 * s2;

  // Constructor with initial values (vector component passed as 3 RbtDouble's)
  // float s1, float vx, float vy, float vz
  *s = c1c2 * c3 - s1s2 * s3;
  v[0] = c1c2 * s3 + s1s2 * c3;
  v[1] = c1 * s2 * c3 - s1 * c2 * s3;
  v[2] = s1 * c2 * c3 + c1 * s2 * s3;
}

void multiplyQuat(float* s, float* v, float* s_original, float* v_original) {
    // Constructor with float and vector (float, vector)

    float s_new = (*s) * (*s_original) - dotProduct3((float*)v, (float*)v_original);
    
    float v_p1[3];
    multiplyVectorByScalar3((float*)v_original, (*s), (float*)v_p1);
    
    float v_p2[3];
    multiplyVectorByScalar3((float*)v, (*s_original), (float*)v_p2);
    
    float v_cross[3];
    crossProduct3((float*)v, (float*)v_original, (float*)v_cross);
    
    v_original[0] = v_p1[0] + v_p2[0] + v_cross[0];
    v_original[1] = v_p1[1] + v_p2[1] + v_cross[1];
    v_original[2] = v_p1[2] + v_p2[2] + v_cross[2];
    *s_original = s_new;

}

void multiplyQuatResult(float* s, float* v, float* s_original, float* v_original, float* s_result, float* v_result) {
    // Constructor with float and vector (float, vector)

    float s_new = (*s) * (*s_original) - dotProduct3((float*)v, (float*)v_original);
    
    float v_p1[3];
    multiplyVectorByScalar3((float*)v_original, (*s), (float*)v_p1);
    
    float v_p2[3];
    multiplyVectorByScalar3((float*)v, (*s_original), (float*)v_p2);
    
    float v_cross[3];
    crossProduct3((float*)v, (float*)v_original, (float*)v_cross);
    
    v_result[0] = v_p1[0] + v_p2[0] + v_cross[0];
    v_result[1] = v_p1[1] + v_p2[1] + v_cross[1];
    v_result[2] = v_p1[2] + v_p2[2] + v_cross[2];
    *s_result = s_new;

}

void FromQuat(float* s, float* v, float* euler) {
float test = (v[0] * v[2]) + (v[1] * (*s));
  if (test > 0.499999f) { // singularity at north pole
    euler[0] = 2.0f * atan2(v[0], (*s));
    euler[1] = M_PI / 2.0f;
    euler[2] = 0.0f;
  } else if (test < -0.499999f) { // singularity at south pole
    euler[0] = -2.0f * atan2(v[0], (*s));
    euler[1] = -M_PI / 2.0f;
    euler[2] = 0.0f;
  } else {
    float sq[3];
    multiply2Vectors3((float*)v, (float*)v, (float*)sq);

    euler[0] = atan2(2.0f * (v[2] * (*s) - v[0] * v[1]),
                           1.0f - 2.0f * (sq[2] + sq[1]));
    euler[1] = asin(2.0f * test);
    euler[2] = atan2(2.0f * (v[0] * (*s) - v[2] * v[1]),
                        1.0f - 2.0f * (sq[0] + sq[1]));
  }
}

void ConjQuat(float* s, float* v, float* s_conj, float* v_conj) {
    // Returns conjugate
    *s_conj = *s;
    v_conj[0] = -v[0];
    v_conj[1] = -v[1];
    v_conj[2] = -v[2];
}

void Rotate(float* chromosomeOrientation, float* axis, float theta) {
    float s;
    float v[3];

    float s_original;
    float v_original[3];

    // a. get Quat for random axis
    RbtQuat((float*)axis, theta, &s, (float*)v);// s and v variable
    // b. get Quat for  initial orientation
    ToQuat((float*)chromosomeOrientation, &s_original, (float*)v_original);
    // c. multiply q * ToQuat()  Multiplication (non-commutative)
    multiplyQuat(&s, (float*)v, &s_original, (float*)v_original);
    // d. get back euler
    FromQuat(&s_original, (float*)v_original, (float*)chromosomeOrientation);
}

void RotateUsingQuat(float* s, float* v, float* w, float* result) {
  // 1. Part: (s * s - v.Dot(v)) * w
  float wScalar = ( (*s) * (*s) ) - dotProduct3((float*)v, (float*)v);
  float wScaled[3];
  multiplyVectorByScalar3((float*)w, wScalar, (float*)wScaled);
  // 2. Part: 2 * s * v.Cross(w)
  float ss = 2.0f * (*s);
  float vCrossW[3];
  crossProduct3((float*)v, (float*)w, (float*)vCrossW);
  float vCrossWscaled[3];
  multiplyVectorByScalar3((float*)vCrossW, ss, (float*)vCrossWscaled);
  // 3. Part: 2 * v * v.Dot(w);
  float vv[3];
  multiplyVectorByScalar3((float*)v, 2.0f, (float*)vv);
  float vDotW = dotProduct3((float*)v, (float*)w);
  float vScaled[3];
  multiplyVectorByScalar3((float*)vv, vDotW, (float*)vScaled);

  // Result:
  result[0] = wScaled[0] + vCrossWscaled[0] + vScaled[0];
  result[1] = wScaled[1] + vCrossWscaled[1] + vScaled[1];
  result[2] = wScaled[2] + vCrossWscaled[2] + vScaled[2];
}

void GetQuatFromAlignVectors(float* v, float* ref, float* q_s, float* q_v) {
  // Set Default:
  *q_s = 1.0f;
  q_v[0] = 0.0f;
  q_v[1] = 0.0f;
  q_v[2] = 0.0f;
  
  // Unitise the two vectors
  float len = lengthNorm3(v);
  float refLen = lengthNorm3(ref);
  if ((len < 0.001f) || (refLen < 0.001f)) {
    // RbtBadArgument
    printf("[Quat] [GetQuatFromAlignVectors] [Zero length vector (v or ref)]\n");
  }
  float vUnit[3];
  float refUnit[3];
  divideVectorByScalar3(v, len, (float*)vUnit);
  divideVectorByScalar3(ref, refLen, (float*)refUnit);
  // Determine the rotation axis and angle needed to overlay the two vectors
  float axis[3];
  crossProduct3((float*)vUnit, (float*)refUnit, (float*)axis);
  // DM 15 March 2006: check for zero-length rotation axis
  // This indicates the vectors are already aligned
  float axisLen = lengthNorm3((float*)axis);
  if (axisLen > 0.001f) {
    float cosPhi = dotProduct3((float*)vUnit, (float*)refUnit);
    if (cosPhi < -1.0f) {
      cosPhi = -1.0f;
    } else if (cosPhi > 1.0f) {
      cosPhi = 1.0f;
    }
    //errno = 0;
    // Convert rotation axis and angle to a quaternion
    float halfPhi = 0.5f * acos(cosPhi);
    if ((halfPhi > 0.001f)) { // && (errno != EDOM)
      float axisUnit[3];
      divideVectorByScalar3((float*)axis, axisLen, (float*)axisUnit);
      // Set return Quat
      *q_s = cos(halfPhi);
      multiplyVectorByScalar3((float*)axisUnit, sin(halfPhi), q_v);
    }

  }

}

void GetQuatFromAlignAxes(PrincipalAxesSyncGPU* prAxes, PrincipalAxesSyncGPU* refAxes, float* s, float* v) {

  // 1) Determine the quaternion needed to align axis1 with reference
  float q1_v[3];
  float q1_s;
  GetQuatFromAlignVectors((float*)prAxes->axis1, (float*)refAxes->axis1, &q1_s, (float*)q1_v);
  // 2) Apply the transformation to axis2
  float axis2[3];
  RotateUsingQuat(&q1_s, (float*)q1_v, (float*)prAxes->axis2, (float*)axis2);
  // 3) Determine the quaternion needed to align axis2 with reference
  float q2_v[3];
  float q2_s;
  GetQuatFromAlignVectors((float*)axis2, (float*)refAxes->axis2, &q2_s, (float*)q2_v);
  // Return the quaternion product (equivalent to both transformations combined)
  multiplyQuatResult(&q2_s, (float*)q2_v, &q1_s, (float*)q1_v, s, v);// q2 * q1
}

#endif
