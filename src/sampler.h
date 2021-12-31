#ifndef SAMPLERH
#define SAMPLERH

class sampler {
    public:
        __device__ sampler() {}
        __device__ virtual vec3 get_sample(curandState *local_rand_state, double* pdf) const = 0;

};

class unit_sphere_sampler: public sampler {
    public:
        __device__ unit_sphere_sampler() {}
        __device__ virtual vec3 get_sample(curandState *local_rand_state, double* pdf) const;
};

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 unit_sphere_sampler::get_sample(curandState *local_rand_state, double* pdf) const {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return unit_vector(p);
}

class cosing_weighted_sampler: public sampler {
    public:
        __device__ cosing_weighted_sampler() {}
        __device__ virtual vec3 get_sample(curandState *local_rand_state, double* pdf) const;
};

__device__ vec3 cosing_weighted_sampler::get_sample(curandState *local_rand_state, double* pdf) const {
    double Xi1 = RANDVEC3.x();
    double Xi2 = RANDVEC3.y();

    double r = sqrt(Xi1);
    double theta = 2. * CUDART_PI_F * Xi2;
    *pdf = sqrt(1-Xi1) / CUDART_PI_F;
    return vec3(r*cos(theta), r*sin(theta), sqrt(1-Xi1));
}

#endif