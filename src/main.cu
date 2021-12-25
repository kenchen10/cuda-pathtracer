#include <iostream>
#include <fstream>
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
#include "light.h"
#include "triangle.h"
#include "utils/OBJ_Loader.h"

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

__device__ vec3 global_illumination(const ray& r, hitable **world, curandState *local_rand_state) {
   ray cur_ray = r;
   vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
   unit_sphere_sampler sampler;
   vec3 reflectance = vec3(1., 1., 1.);
   vec3 f = reflectance / CUDART_PI_F;
   for(int i = 0; i < r.depth; i++) {
      hit_record rec;
      if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
        vec3 target;
        double pdf;
        vec3 f = rec.BSDF->evaluate(r.direction(), &target, rec.p, rec.normal, &pdf, local_rand_state);
        cur_attenuation *= f;
        cur_ray = ray(rec.p, target-rec.p);
      }
      else {
            if (i > 0) {
                if (i == 1 && rec.BSDF->reflective()) {
                    return vec3(0., 0., 0.);
                }
                vec3 unit_direction = unit_vector(cur_ray.direction());
                point_light pt_l = point_light(vec3(1., 1., 1.), vec3(0., 0, 2));
                vec3 light_dir;
                double light_dist;
                double pdf;
                vec3 wi;
                vec3 light_radiance = pt_l.sample_light(rec.p, &light_dir, &light_dist, &pdf);
                ray shadow_ray = ray(rec.p, light_dir);
                if (!(*world)->hit(shadow_ray, 0.001f, FLT_MAX, rec)) {
                    double cos = abs(light_dir.z());
                    return cur_attenuation * light_radiance;
                }
            }
            else {
                return vec3(0., 0., 0.);
            }
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

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, int max_depth, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    hit_record rec;
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v);
        r.depth = max_depth;
        col += global_illumination(r, world, &local_rand_state);
    }
    fb[pixel_index] = col/float(ns);
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, vec3 *vec_list, int num_prims) {
    diffuse *red = new diffuse(vec3(1.f, 0.f, 0.f));
    emissive *e = new emissive(vec3(1., 1., 1.), vec3(0., 0., 0.));
    mirror *m = new mirror(vec3(1.f, 1.f, 1.f));
    diffuse *green = new diffuse(vec3(0.f, 1.f, 0.f));
    diffuse *blue = new diffuse(vec3(0.f, 0.f, 1.f));
    diffuse *yellow = new diffuse(vec3(1.f, 1.f, 0.f));
    diffuse *white = new diffuse(vec3(1.f, 1.f, 1.f));
    diffuse *p = new diffuse(vec3(1.f, 0.f, 1.f) * 2.);
    // Image
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 1900;
    vec3 lookfrom(0,0,10);
    vec3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // *(d_list+0)   = new sphere(vec3(0,-.6,-.5), 0.3, m);
        // *(d_list+1) = new sphere(vec3(0,0,-.5), 0.1, green);
        // *(d_list+2) = new sphere(vec3(.5,-.6,-.5), 0.2, p);
        // *(d_list+3) = new sphere(vec3(-.6,-.6,-.5), 0.23, red);
        // *(d_list+4) = new triangle(vec3(-1,-1,-3), vec3(-1, 1, -3), vec3(1, 1, -3), white);
        // *(d_list+5) = new triangle(vec3(1, 1, -3), vec3(-1,-1,-3), vec3(1, -1, -3), white);
        // *(d_list+6) = new triangle(vec3(-1,-1,-3), vec3(1, -1, -3), vec3(1, -1, 0), white);
        // *(d_list+7) = new triangle(vec3(-1,-1,-3), vec3(-1, -1, 0), vec3(1, -1, 0), white);
        // *(d_list+8) = new triangle(vec3(-1,-1,-3), vec3(-1, 1, -3), vec3(-1, 1, 0), white);
        // *(d_list+9) = new triangle(vec3(-1,-1,-3), vec3(-1, -1, 0), vec3(-1, 1, 0), white);
        // *(d_list+10) = new triangle(vec3(1,-1,-3), vec3(1, 1, -3), vec3(1, 1, 0), white);
        // *(d_list+11) = new triangle(vec3(1,-1,-3), vec3(1, -1, 0), vec3(1, 1, 0), white);
        // *(d_list+12) = new triangle(vec3(-1,1,-3), vec3(1, 1, -3), vec3(1, 1, 0), white);
        // *(d_list+13) = new triangle(vec3(-1,1,-3), vec3(-1, 1, 0), vec3(1, 1, 0), white);
        for (int i = 0; i < 4000; i++)
        {
            *(d_list+i) = new triangle(vec_list[3*i], vec_list[3*i+1], vec_list[3*i+2], yellow);
        }
        // *(d_list+14) = new triangle(vec_list[0], vec_list[1], vec_list[2], yellow);
        // *(d_list+15) = new triangle(vec_list[3], vec_list[4], vec_list[5], yellow);
        // *(d_list+16) = new triangle(vec_list[6], vec_list[7], vec_list[8], yellow);
        // *(d_list+17) = new triangle(vec_list[9], vec_list[10], vec_list[11], yellow);
        *d_world    = new hitable_list(d_list,num_prims);
        *d_camera   = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera, int num_prims) {
    for (int i = 0; i < num_prims; i++) {
        delete *(d_list+i);
    }
    // delete *(d_list);
    // delete *(d_list+1);
    // delete *(d_list+2);
    // delete *(d_list+3);
    // delete *(d_list+4);
    // delete *(d_list+5);
    // delete *(d_list+6);
    // delete *(d_list+7);
    // delete *(d_list+8);
    // delete *(d_list+9);
    // delete *(d_list+10);
    // delete *(d_list+11);
    // delete *(d_list+12);
    // delete *(d_list+13);
    // delete *(d_list+14);
    delete *d_world;
    delete *d_camera;
}

int main() {
	// Load .obj File
    // load_mesh();
    int nx = 1200;
    int ny = 600;
    int ns = 10;
    int tx = 16;
    int ty = 16;
    int max_depth = 2;
    int num_prims = 0 + 4000;

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

    // Load mesh
    vec3 *vec_list;
    checkCudaErrors(cudaMallocManaged((void **)&vec_list, 108*sizeof(vec3)));
    objl::Loader Loader;
	// Load .obj File
	bool loadout = Loader.LoadFile("/home/kenny/Documents/Projects/cuda-rt/meshes/obj/bunny.obj");

	// Check to see if it loaded
    int count = 0;
	// If so continue
	if (loadout)
	{
        // Create/Open e1Out.txt
        std::cerr << Loader.LoadedMeshes.size() << "\n";
		// Go through each loaded mesh and out its contents
		for (int i = 0; i < Loader.LoadedMeshes.size(); i++)
		{
			// Copy one of the loaded meshes to be our current mesh
			objl::Mesh curMesh = Loader.LoadedMeshes[i];
            std::cerr << curMesh.Indices.size() << "\n";
            for (int j = 0; j < curMesh.Indices.size(); j += 3)
			{
				int i1 = curMesh.Indices[j];
                int i2 = curMesh.Indices[j + 1];
                int i3 = curMesh.Indices[j + 2];
                // std::cerr << i1 << " " << i2 << " " << i3 << "\n";
                vec_list[count] = vec3(curMesh.Vertices[i1].Position.X, curMesh.Vertices[i1].Position.Y, curMesh.Vertices[i1].Position.Z);
                vec_list[count + 1] = vec3(curMesh.Vertices[i2].Position.X, curMesh.Vertices[i2].Position.Y, curMesh.Vertices[i2].Position.Z);
                vec_list[count + 2] = vec3(curMesh.Vertices[i3].Position.X, curMesh.Vertices[i3].Position.Y, curMesh.Vertices[i3].Position.Z);
                count += 3;
                // vec_list[count].print();
                // vec_list[count+1].print();
                // vec_list[count+2].print();
			}
        }

        std::cerr << count << "\n";
    }

    // make our world of hitables & the camera
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_prims*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    // print(vec_list[0]);print(vec_list[1]);print(vec_list[2]);
    create_world<<<1,1>>>(d_list,d_world,d_camera,vec_list,num_prims);
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
    render<<<blocks, threads>>>(fb, nx, ny,  ns, max_depth, d_camera, d_world, d_rand_state);
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
    free_world<<<1,1>>>(d_list,d_world,d_camera,num_prims);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(vec_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}