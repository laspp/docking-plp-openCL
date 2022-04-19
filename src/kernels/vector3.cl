#ifndef VECTOR3_CL_H
#define VECTOR3_CL_H

#include <tyche_i.cl>

void randomUnitVector3(float* vector, tyche_i_state* state) {
    vector[2] = 2.0f * tyche_i_float(state[0]) - 1.0f;
    float t = 2.0f * M_PI * tyche_i_float(state[0]);
    float w = sqrt(1.0f - vector[2] * vector[2]);
    vector[0] = w * cos(t);
    vector[1] = w * sin(t);
}

void unitVector3(float* vector, float* unit, float length) {
    if(length > 0.0f) {
        unit[0] = vector[0] / length;
        unit[1] = vector[1] / length;
        unit[2] = vector[2] / length;
    } else {
        unit[0] = vector[0];
        unit[1] = vector[1];
        unit[2] = vector[2];
    }
}

void saveVector3(float* what, float* to) {
    to[0] = what[0];
    to[1] = what[1];
    to[2] = what[2];
}

float lengthNorm3(float* vector) {
    //xyz.norm() Returns magnitude of vector (or distance from origin)
   return sqrt(vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]);
}

float dotProduct3(float* v1, float* v2) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

void crossProduct3(float* v1, float* v2, float* result) {
    result[0] = v1[1]*v2[2] - v1[2]*v2[1];
    result[1] = v1[2]*v2[0] - v1[0]*v2[2];
    result[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

void multiplyVectorByScalar3(float* v, float s, float* result) {
    result[0] = v[0] * s;
    result[1] = v[1] * s;
    result[2] = v[2] * s;
}

void multiply2Vectors3(float* v1, float* v2, float* result) {
    // Scalar product (coord * coord : component-wise multiplication) RESULT VECTOR!
    result[0] = v1[0] * v2[0];
    result[1] = v1[1] * v2[1];
    result[2] = v1[2] * v2[2];
}

void divideVectorByScalar3(float* v, float scalar, float* result) {
    result[0] = v[0] / scalar;
    result[1] = v[1] / scalar;
    result[2] = v[2] / scalar;
}

void negateVector3(float* v, float* result) {
    result[0] = -v[0];
    result[1] = -v[1];
    result[2] = -v[2];
}

void subtract2Vectors3(float* v1, float* v2, float* result) {
    result[0] = v1[0] - v2[0];
    result[1] = v1[1] - v2[1];
    result[2] = v1[2] - v2[2];
}

void add2Vectors3(float* v1, float* v2, float* result) {
    result[0] = v1[0] + v2[0];
    result[1] = v1[1] + v2[1];
    result[2] = v1[2] + v2[2];
}

void zerosNxN(float* matrix, int n) {
    int i;
    for(i = 0; i < n*n; i++) {
        matrix[i] = 0.0f;
    }
}

// Returns distance between two coords
float distance2Points3(float* v1, float* v2) {
    float difference[3];
    subtract2Vectors3(v2, v1, (float*)difference);
    return lengthNorm3((float*)difference);
}

// 1 = VALID, 0 = NOT VALID
int isPointInsideCuboid3(float* p, float* min, float* max) {
    return ( p[0] >=min[0] && p[0] <= max[0] &&
             p[1] >=min[1] && p[1] <= max[1] &&
             p[2] >=min[2] && p[2] <= max[2]);
}

#endif
