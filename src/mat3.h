#ifndef MAT3H
#define MAT3H

#include "vec3.h"

class mat3 {
    public:
        __device__ mat3(const vec3 a, const vec3 b, const vec3 c) : v1(a), v2(b), v3(c) {}
        vec3 v1, v2, v3;
}

#endif