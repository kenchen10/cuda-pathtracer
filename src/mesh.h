#ifndef MESHH
#define MESHH

class mesh
{
    public:
        __device__ mesh(vec3** v) : vec_list(v) {}
        __device__ void get_tris(hitable **d_list, int vec_count) const {
            diffuse *white = new diffuse(vec3(1.f, 1.f, 1.f));
            for(int i = 0; i < (int)(vec_count/3); i++) 
            {
                vec3* v1 = vec_list[i*3];
                vec3* v2 = vec_list[i*3+1];
                vec3* v3 = vec_list[i*3+2];
                *(d_list + i) = new triangle(*v1, *v2, *v3, white);
            }
        }

        vec3 **vec_list;
};

#endif