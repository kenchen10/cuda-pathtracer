#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include "vec3.h"
#include "ray.h"
#include "bsdf.h"
#include "sphere.h"
#include "hitable.h"
#include "hitable_list.h"
#include "camera.h"
#include "sampler.h"


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
   ray cur_ray = r;
   vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
   unit_sphere_sampler sampler;
   // Temporary variables
   vec3 reflectance = vec3(1., 1., 1.);
   vec3 f = reflectance / CUDART_PI_F;
   for(int i = 0; i < 50; i++) {
      hit_record rec;
      if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
        vec3 target;
        double pdf;
        vec3 f = rec.BSDF->evaluate(r.direction(), &target, rec.p, rec.normal, &pdf, local_rand_state);
        // vec3 target = cur_ray.direction() - 2.f * dot(cur_ray.direction(), rec.normal) * rec.normal;
        cur_attenuation *= f;
        cur_ray = ray(rec.p, target-rec.p);
      }
      else {
           vec3 unit_direction = unit_vector(cur_ray.direction());
           float t = 0.5f*(unit_direction.y() + 1.0f);
           vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
           return cur_attenuation * c;
        }
      }
   return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v);
        col += color(r, world, &local_rand_state);
    }
    fb[pixel_index] = col/float(ns);
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    diffuse *red = new diffuse(vec3(1.f, 0.f, 0.f));
    diffuse *green = new diffuse(vec3(0.f, 1.f, 0.f));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5, green);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, red);
        *d_world    = new hitable_list(d_list,2);
        *d_camera   = new camera();
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
    delete *d_camera;
}

int main() {
    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    // make our world of hitables & the camera
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}