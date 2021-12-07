#ifndef BSDFH
#define BSDFH
#include "sampler.h"

class bsdf {
    public:
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi) = 0;
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) = 0;
        __device__ virtual vec3 emission(vec3 wo) const = 0;
};

class diffuse : public bsdf {
    public:
        __device__ diffuse(const vec3 r): attenuation(r) {}
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi);
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state);
        __device__ virtual vec3 emission(vec3 wo) const { return vec3(0.f, 0.f, 0.f); };
    
    private:
        vec3 attenuation;
        unit_sphere_sampler sampler;
};

__device__ vec3 diffuse::f(const vec3 wo, const vec3 wi) {
    return attenuation / CUDART_PI_F;
}

__device__ vec3 diffuse::evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) {
    *wi = p + n + sampler.get_sample(local_rand_state);
    return f(wo, *wi);
}

class mirror : public bsdf {
    public:
        __device__ mirror(const vec3 r): attenuation(r) {}
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi);
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state);
        __device__ virtual vec3 emission(vec3 wo) const { return vec3(0.f, 0.f, 0.f); };
    
    private:
        vec3 attenuation;
        unit_sphere_sampler sampler;
};

__device__ vec3 mirror::f(const vec3 wo, const vec3 wi) {
    return vec3(1.f, 1.f, 1.f);
}

__device__ vec3 mirror::evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) {
    *wi = wo - 2.f * dot(wo, n) * n;
    return attenuation;
}

#endif