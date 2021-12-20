#ifndef TRIANGLEH
#define TRIANGLEH

#include "hitable.h"
#include "bsdf.h"

class triangle: public hitable  {
    public:
        __device__ triangle() {}
        __device__ triangle(vec3 a, vec3 b, vec3 c, bsdf *bs) : p1(a), p2(b), p3(c), BSDF(bs) {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 p1, p2, p3;
        __device__ virtual vec3 calc_n() {
            vec3 a = p2 - p1;
            vec3 b = p3 - p1;
            return vec3(a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(), a.x() * b.y() - a.y() * b.x());
        }
        vec3 n = calc_n();
        bsdf *BSDF;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 E1 = p2 - p1;
    vec3 E2 = p3 - p1;
    vec3 S = r.origin() - p1;
    vec3 S1 = cross(r.direction(), E2);
    vec3 S2 = cross(S, E1);
    float denom = dot(S1, E1);
    float t = dot(S2, E2) / denom;
    float b1 = dot(S1, S) / denom;
    float b2 = dot(S2, r.direction()) / denom;
    float sub = 1 - b1 - b2;
    if (t <= t_max && t >= t_min && b1 >= 0. && b1 <= 1. && b2 >= 0. && b2 <= 1. && sub >= 0. && sub <= 1.) {
        rec.t = t;
        rec.p = r.at(rec.t);
        rec.normal = unit_vector(sub * n + b1 * n + b2 * n);
        rec.BSDF = BSDF;
        return true;
    }
    return false;
}


#endif