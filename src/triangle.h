#ifndef TRIANGLEH
#define TRIANGLEH

#include "hitable.h"
#include "bsdf.h"

class triangle: public hitable  {
    public:
        __device__ triangle() {}
        __device__ triangle(vec3 a, vec3 b, vec3 c, bsdf *b) : p1(a), p2(b), p3(c), BSDF(b) {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 p1, p2, p3;
        vec3 n1, n2, n3;
        bsdf *BSDF;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    
    return false;
}


#endif