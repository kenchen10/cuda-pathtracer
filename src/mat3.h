#ifndef MAT3H
#define MAT3H

#include "vec3.h"

class mat3 {
    public:
        __host__ __device__ mat3() { v1 = vec3(0, 0, 0); v2 = vec3(0, 0, 0); v3 = vec3(0, 0, 0); }
        __host__ __device__ mat3(const vec3 a, const vec3 b, const vec3 c) : v1(a), v2(b), v3(c) {}
        __host__ __device__ inline mat3 T() const { return mat3(vec3(v1.x(), v2.x(), v3.x()), vec3(v1.y(), v2.y(), v3.y()), vec3(v1.z(), v2.z(), v3.z())); }
        vec3 v1, v2, v3;
};

__host__ __device__ inline vec3 operator*(const mat3 &m, const vec3 &v) {
    return vec3(dot(m.v1, v), dot(m.v2, v), dot(m.v3, v));
}

#endif