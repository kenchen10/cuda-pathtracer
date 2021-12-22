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
                point_light pt_l = point_light(vec3(1., 1., 1.), vec3(0., .8, 0));
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

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
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
    vec3 lookfrom(0,0,5);
    vec3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list+0)   = new sphere(vec3(0,-.6,-.5), 0.3, m);
        *(d_list+1) = new sphere(vec3(0,0,-.5), 0.1, green);
        *(d_list+2) = new sphere(vec3(.5,-.6,-.5), 0.2, blue);
        *(d_list+3) = new sphere(vec3(-.6,-.6,-.5), 0.23, yellow);
        // *(d_list+4) = new sphere(vec3(0,-10.5,-1), 10, p);
        // *(d_list+5) = new triangle(vec3(0,-.6,-3), vec3(0, 3, -3), vec3(1, -.5, -3), m);
        *(d_list+4) = new triangle(vec3(-1,-1,-3), vec3(-1, 1, -3), vec3(1, 1, -3), white);
        *(d_list+5) = new triangle(vec3(1, 1, -3), vec3(-1,-1,-3), vec3(1, -1, -3), white);
        // *(d_list+6) = new triangle(vec3(-1,1,-3), vec3(1, 1, -3), vec3(1, 1, 0), white);
        // *(d_list+7) = new triangle(vec3(-1,1,-3), vec3(-1, 1, 0), vec3(1, 1, 0), white);
        *(d_list+6) = new triangle(vec3(-1,-1,-3), vec3(1, -1, -3), vec3(1, -1, 0), white);
        *(d_list+7) = new triangle(vec3(-1,-1,-3), vec3(-1, -1, 0), vec3(1, -1, 0), white);
        *(d_list+8) = new triangle(vec3(-1,-1,-3), vec3(-1, 1, -3), vec3(-1, 1, 0), white);
        *(d_list+9) = new triangle(vec3(-1,-1,-3), vec3(-1, -1, 0), vec3(-1, 1, 0), white);
        *(d_list+10) = new triangle(vec3(1,-1,-3), vec3(1, 1, -3), vec3(1, 1, 0), white);
        *(d_list+11) = new triangle(vec3(1,-1,-3), vec3(1, -1, 0), vec3(1, 1, 0), white);
        *(d_list+12) = new triangle(vec3(-1,1,-3), vec3(1, 1, -3), vec3(1, 1, 0), white);
        *(d_list+13) = new triangle(vec3(-1,1,-3), vec3(-1, 1, 0), vec3(1, 1, 0), white);
        *d_world    = new hitable_list(d_list,14);
        *d_camera   = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    delete *(d_list);
    delete *(d_list+1);
    delete *(d_list+2);
    delete *(d_list+3);
    delete *(d_list+4);
    delete *(d_list+5);
    delete *(d_list+6);
    delete *(d_list+7);
    delete *(d_list+8);
    delete *(d_list+9);
    delete *(d_list+10);
    delete *(d_list+11);
    delete *(d_list+12);
    delete *(d_list+13);
    delete *d_world;
    delete *d_camera;
}

int load_mesh() {
    // Initialize Loader
	objl::Loader Loader;

	// Load .obj File
	bool loadout = Loader.LoadFile("/home/kenny/Documents/Projects/cuda-rt/meshes/obj/box_stack.obj");

	// Check to see if it loaded
    std::ofstream file("e1Out.txt");
	// If so continue
	if (loadout)
	{
        // Create/Open e1Out.txt

		// Go through each loaded mesh and out its contents
		for (int i = 0; i < Loader.LoadedMeshes.size(); i++)
		{
			// Copy one of the loaded meshes to be our current mesh
			objl::Mesh curMesh = Loader.LoadedMeshes[i];

			// Print Mesh Name
			file << "Mesh " << i << ": " << curMesh.MeshName << "\n";

			// Print Vertices
			file << "Vertices:\n";

			// Go through each vertex and print its number,
			//  position, normal, and texture coordinate
			for (int j = 0; j < curMesh.Vertices.size(); j++)
			{
				file << "V" << j << ": " <<
					"P(" << curMesh.Vertices[j].Position.X << ", " << curMesh.Vertices[j].Position.Y << ", " << curMesh.Vertices[j].Position.Z << ") " <<
					"N(" << curMesh.Vertices[j].Normal.X << ", " << curMesh.Vertices[j].Normal.Y << ", " << curMesh.Vertices[j].Normal.Z << ") " <<
					"TC(" << curMesh.Vertices[j].TextureCoordinate.X << ", " << curMesh.Vertices[j].TextureCoordinate.Y << ")\n";
			}

			// Print Indices
			file << "Indices:\n";

			// Go through every 3rd index and print the
			//	triangle that these indices represent
			for (int j = 0; j < curMesh.Indices.size(); j += 3)
			{
				file << "T" << j / 3 << ": " << curMesh.Indices[j] << ", " << curMesh.Indices[j + 1] << ", " << curMesh.Indices[j + 2] << "\n";
			}

			// Print Material
			file << "Material: " << curMesh.MeshMaterial.name << "\n";
			file << "Ambient Color: " << curMesh.MeshMaterial.Ka.X << ", " << curMesh.MeshMaterial.Ka.Y << ", " << curMesh.MeshMaterial.Ka.Z << "\n";
			file << "Diffuse Color: " << curMesh.MeshMaterial.Kd.X << ", " << curMesh.MeshMaterial.Kd.Y << ", " << curMesh.MeshMaterial.Kd.Z << "\n";
			file << "Specular Color: " << curMesh.MeshMaterial.Ks.X << ", " << curMesh.MeshMaterial.Ks.Y << ", " << curMesh.MeshMaterial.Ks.Z << "\n";
			file << "Specular Exponent: " << curMesh.MeshMaterial.Ns << "\n";
			file << "Optical Density: " << curMesh.MeshMaterial.Ni << "\n";
			file << "Dissolve: " << curMesh.MeshMaterial.d << "\n";
			file << "Illumination: " << curMesh.MeshMaterial.illum << "\n";
			file << "Ambient Texture Map: " << curMesh.MeshMaterial.map_Ka << "\n";
			file << "Diffuse Texture Map: " << curMesh.MeshMaterial.map_Kd << "\n";
			file << "Specular Texture Map: " << curMesh.MeshMaterial.map_Ks << "\n";
			file << "Alpha Texture Map: " << curMesh.MeshMaterial.map_d << "\n";
			file << "Bump Map: " << curMesh.MeshMaterial.map_bump << "\n";

			// Leave a space to separate from the next mesh
			file << "\n";
		}

		// Close File
		file.close();
	}
	// If not output an error
	else
	{
		// Output Error
		std::cerr << "Failed to Load File. May have failed to find it or it was not an .obj file.\n";

		// Close File
		file.close();
	}

	// Exit the program
	return 0;

}

int main() {
	// Load .obj File
    // load_mesh();
    int nx = 1200;
    int ny = 600;
    int ns = 50;
    int tx = 16;
    int ty = 16;
    int max_depth = 15;

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
    checkCudaErrors(cudaMalloc((void **)&d_list, 14*sizeof(hitable *)));
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