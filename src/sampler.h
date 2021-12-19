#ifndef SAMPLERH
#define SAMPLERH

class sampler {
    public:
        __device__ sampler() {}
        __device__ virtual vec3 get_sample(curandState *local_rand_state) const = 0;

};

class unit_sphere_sampler: public sampler {
    public:
        __device__ unit_sphere_sampler() {}
        __device__ virtual vec3 get_sample(curandState *local_rand_state) const;
};

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 unit_sphere_sampler::get_sample(curandState *local_rand_state) const {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return unit_vector(p);
}

#endif