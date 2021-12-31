#ifndef PLANEH
#define PLANEH

#include "hitable.h"
#include "bsdf.h"
#include "triangle.h"

class plane: public hitable  {
    public:
        __device__ plane() {}
        __device__ plane(vec3 a, vec3 b, bsdf *bs) : p1(a), p2(b), BSDF(bs) {
            
        };
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 p1, p2;
        vec3 n;
        bsdf *BSDF;
};

__device__ bool plane::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return false;
}

#endif