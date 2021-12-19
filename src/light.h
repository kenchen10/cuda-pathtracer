#ifndef LIGHTH
#define LIGHTH

#include "vec3.h"
#include "bsdf.h"

class light {
    public:
        __device__ virtual vec3 sample_light(const vec3 p, vec3* wi, double* d, double* pdf) = 0;
    
        vec3 radiance;
        bsdf *BSDF;
};

class point_light : light {
    public:
        __device__ point_light(const vec3 r, const vec3 p): radiance(r), pos(p) {}
        __device__ virtual vec3 sample_light(const vec3 p, vec3* wi, double* d, double* pdf);

        vec3 radiance;
        vec3 pos;
        bsdf *BSDF;
};

__device__ vec3 point_light::sample_light(const vec3 p, vec3* wi, double* d, double* pdf) {
    vec3 dir = pos - p;
    *wi = unit_vector(dir);
    *d = dir.length();
    *pdf = 1.0;
    return radiance;
}

#endif