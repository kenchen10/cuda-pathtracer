#ifndef BSDFH
#define BSDFH
#include "sampler.h"

class bsdf {
    public:
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi) = 0;
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) = 0;
        __device__ virtual vec3 emission(vec3 wo) const = 0;
        __device__ virtual bool reflective() const = 0;
};

class diffuse : public bsdf {
    public:
        __device__ diffuse(const vec3 r): attenuation(r) {}
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi);
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state);
        __device__ virtual vec3 emission(vec3 wo) const { return vec3(0.f, 0.f, 0.f); };
        __device__ virtual bool reflective() const { return false; };
    
    private:
        vec3 attenuation;
        unit_sphere_sampler sampler;
};

__device__ vec3 diffuse::f(const vec3 wo, const vec3 wi) {
    return attenuation / CUDART_PI_F;
}

__device__ vec3 diffuse::evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) {
    *wi = unit_vector(p + n + sampler.get_sample(local_rand_state));
    *pdf = CUDART_PI_F;
    return f(wo, *wi);
}

class mirror : public bsdf {
    public:
        __device__ mirror(const vec3 r): attenuation(r) {}
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi);
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state);
        __device__ virtual vec3 emission(vec3 wo) const { return vec3(0.f, 0.f, 0.f); };
        __device__ virtual bool reflective() const { return true; };
    
    private:
        vec3 attenuation;
        unit_sphere_sampler sampler;
};

__device__ vec3 mirror::f(const vec3 wo, const vec3 wi) {
    return vec3(1.f, 1.f, 1.f);
}

__device__ vec3 mirror::evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) {
    *wi = reflect(wo, n);
    *pdf = 1.;
    return attenuation;
}

class glass : public bsdf {
    public:
        __device__ glass(const vec3 t, const vec3 r, double rough, double ior): transmittance(t), attenuation(r), roughness(rough), ior(ior) {}
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi);
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state);
        __device__ virtual vec3 emission(vec3 wo) const { return vec3(0.f, 0.f, 0.f); };
        __device__ virtual bool reflective() const { return true; };
    
    private:
        vec3 attenuation;
        double ior;
        double roughness;
        vec3 transmittance;
};

__device__ vec3 glass::f(const vec3 wo, const vec3 wi) {
    return vec3(1.f, 1.f, 1.f);
}

__device__ vec3 glass::evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) {
    // vec3 n = vec3(0, 0, 1);
    // float eta = ior / 1.;
    // if (dot(wo, n) > 0) {
    //     eta = 1. / ior;
    // }
    // bool total_internal_reflection = refract(wo, wi, ior);
    // if (!total_internal_reflection) {
    //     reflect(wo, &wi);
    //     *pdf = 1.;
    //     return reflectance / abs(cos(*wi));
    // } 
    // float R0 = pow((1. - ior) / (1. + ior), 2);
    // float R = R0 + (1. - R0) * pow(1. - abs(dot(n, wo)), 5);
    // if (coin_flip(R)) {
    //     reflect(wo, wi);
    //     *pdf = R;
    //     return R * reflectance / abs_cos_theta(*wi);
    // }
    // *pdf = 1. - R;
    // return (1. - R) * transmittance / abs_cos_theta(*wi) * pow(eta, 2);
    return attenuation;
}

class emissive : public bsdf {
    public:
        __device__ emissive(const vec3 r, const vec3 p): attenuation(r), pos(p) {}
        __device__ virtual vec3 f(const vec3 wo, const vec3 wi);
        __device__ virtual vec3 evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state);
        __device__ virtual vec3 emission(vec3 wo) const { 
            if (dot(pos, wo) > 0.) {
                printf("%f", dot(pos, wo));
            }
            return attenuation * max(0.0, dot(pos, wo)); 
        };
        __device__ virtual bool reflective() const { return true; };
    
    private:
        vec3 attenuation;
        unit_sphere_sampler sampler;
        vec3 pos;
};

__device__ vec3 emissive::f(const vec3 wo, const vec3 wi) {
    return vec3(1.f, 1.f, 1.f);
}

__device__ vec3 emissive::evaluate(const vec3 wo, vec3 *wi, vec3 p, vec3 n, double *pdf, curandState *local_rand_state) {
    return attenuation * max(0.0, dot(pos, wo));
}

#endif