#ifndef RANDOM_EXTRA_CL_H
#define RANDOM_EXTRA_CL_H

#include <tyche_i.cl>

int GetRandomInt(int nMax, tyche_i_state* state) {
    //get a random integer between 0 and nMax-1
    int randInt = (int) (tyche_i_float(state[0]) * (float)nMax);
    int check = (randInt >= nMax);
    return (check != 1) * randInt + check * (nMax - 1);
}

float PositiveCauchyRandomWMeanVariance(float mean, float variance, tyche_i_state* state) {

    float t = tan(M_PI * (tyche_i_double(state[0]) - 0.5f));

    t = (t <= 100.0f) * t + (t > 100.0f) * 100.0f;
    t = (t >= -100.0f) * t + (t < -100.0f) * -100.0f;

    return fabs(mean + variance * t);
}

#endif
