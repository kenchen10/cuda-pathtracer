#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

class camera {
    public:
        __device__ camera(
            vec3 lookfrom,
            vec3 lookat,
            vec3   vup,
            double vfov, // vertical field-of-view in degrees
            double aspect_ratio,
            double aperture,
            double focus_dist,
            double _time0 = 0,
            double _time1 = 0
        ) {
            auto theta = 0.017453292 * (vfov);
            auto h = tan(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            lens_radius = aperture / 2;
            time0 = _time0;
            time1 = _time1;
        }
        __device__ ray get_ray(double s, double t) const {

            return ray(
                origin,
                lower_left_corner + s*horizontal + t*vertical - origin
            );
        }

    private:
        vec3 origin;
        vec3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        double lens_radius;
        double time0, time1;  // shutter open/close times
};

#endif