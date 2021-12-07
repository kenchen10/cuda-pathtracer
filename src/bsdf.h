#ifndef BSDFH
#define BSDFH
#include "sampler.h"

class bsdf {
    public:
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi) = 0;
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, hit_record rec, double *pdf, curandState *local_rand_state) = 0;
        __device__ virtual vec3 emission(vec3 wo) const = 0;
};

class diffuse : public bsdf {
    public:
        __device__ diffuse(const vec3 r): attenuation(r) {}
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi) = 0;
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, hit_record rec, double *pdf, curandState *local_rand_state) = 0;
        __device__ virtual vec3 emission(vec3 wo) const = 0;
    
    private:
        vec3 attenuation;
        unit_sphere_sampler sampler;
};

__device__ vec3 diffuse::f(const vec3 wo, const vec3 wi) {
    return attenuation / CUDART_PI_F;
}

__device__ vec3 diffuse::evaluate(const vec3 wo, vec3 *wi, hit_record rec, double *pdf, curandState *local_rand_state) {
    *wi = rec.p + rec.normal + sampler.get_sample(local_rand_state);
    return f(wo, *wi);
}

__device__ vec3 diffuse::emission(const vec3 wo) const {
    return vec3(0.f, 0.f, 0.f);
}

#endif