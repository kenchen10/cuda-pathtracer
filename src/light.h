#ifndef LIGHTH
#define LIGHTH

#include "vec3.h"
#include "bsdf.h"
#include "sampler.h"

class light {
    public:
        __device__ virtual vec3 sample_light(const vec3 p, vec3* wi, double* d, double* pdf, curandState *local_rand_state) = 0;
    
        vec3 radiance;
        bsdf *BSDF;
};

class point_light : light {
    public:
        __device__ point_light(const vec3 r, const vec3 p): radiance(r), pos(p) {}
        __device__ virtual vec3 sample_light(const vec3 p, vec3* wi, double* d, double* pdf, curandState *local_rand_state);

        vec3 radiance;
        vec3 pos;
        bsdf *BSDF;
};

__device__ vec3 point_light::sample_light(const vec3 p, vec3* wi, double* d, double* pdf, curandState *local_rand_state) {
    vec3 dir = pos - p;
    *wi = unit_vector(dir);
    *d = dir.length();
    *pdf = 1.0;
    return radiance;
}

class area_light : light {
    public:
        __device__ area_light(const vec3 r, const vec3 p1, const vec3 p2): radiance(r), pos1(p1), pos2(p2) {}
        __device__ virtual vec3 sample_light(const vec3 p, vec3* wi, double* d, double* pdf, curandState *local_rand_state);

        vec3 radiance;
        vec3 pos1, pos2;
        bsdf *BSDF;
};

__device__ vec3 area_light::sample_light(const vec3 p, vec3* wi, double* d, double* pdf, curandState *local_rand_state) {
    float w = fabs(pos1.x() - pos2.x());
    float h = fabs(pos1.y() - pos2.y());
    float de = fabs(pos1.z() - pos2.z());
    float posx = w * RANDVEC3.x();
    float posy = h * RANDVEC3.y();
    float posz = de * RANDVEC3.z();
    vec3 dir = vec3(posx, posy, posz) - p;
    *wi = unit_vector(dir);
    *d = dir.length();
    *pdf = *d / (w * h);
    return radiance;
}

#endif