// Like volk.cu but with different layers of cells along the z axis
// with rules similar to Volkening (2018) for pigment cell interactions
// Compilation
//
// $ nvcc -std=c++14 -arch=sm_86 {"compiler flags"} Limb_model_simulation.cu
// The values for "-std" and "-arch" flags will depend on your version of CUDA
// and the specific GPU model you have respectively. e.g. -std=c++14 works for
// CUDA version 11.6 and -arch=sm_86 corresponds to the generation of NVIDIA
// Geforce 30XX cards.
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/scan.h>

#include <iterator>
#include <regex>

#include "../include/dmesh.cuh"
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/mesh.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"
#include "../params/params.h"  // load simulation parameters

// Macro that builds the cell variable type - instead of type float3 we are
// making a instance of Cell with attributes x,y,z,u,v where u and v are
// diffusible chemicals
// MAKE_PT(Cell); // float3 i .x .y .z .u .v .whatever
// to use MAKE_PT(Cell) replace every instance of float3 with Cell
// MAKE_PT(Cell);

// define global variables for the GPU
__device__ float* d_mech_str;
__device__ int* d_cell_type; 
__device__ float3* d_W;  // random number from Weiner process for stochasticity
__device__ bool* d_in_slow;   // whether a cell is in a slow region
__device__ Pm d_pm;           // simulation parameters (host h_pm)
__device__ float3 d_tis_min;  // min coordinate of tissue mesh
__device__ float3 d_tis_max;  // max coordinate of tissue mesh
// per-type neighbour counters will be stored as flattened arrays
#define N_TYPES 5
#define NGS_IDX(type,i) (((type) * d_pm.n_max) + (i))
__device__ int* d_ngs_45;
__device__ int* d_ngs_75;
__device__ int* d_ngs_82;
__device__ int* d_ngs_90;   // flattened short-range per-type counters [type*n_max + i]
__device__ int* d_ngs_d; // flattened donut/long-range per-type counters
// Cell types
// -1 = marked for death
// 0 = M
// 1 = Xd
// 2 = Xl
// 3 = Id
// 4 = Il
template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    // counting cells in different nbhds
    // N.B. cells outside the cube size will not be counted!
    // if ((dist > 0.318) and (dist < 0.318 + 0.025)) {  // donut
    if ((dist > 0.21) and (dist < 0.21 + 0.04)) {  // donut
        if (d_cell_type[j] >= 0 && d_cell_type[j] < N_TYPES) {
            atomicAdd(&d_ngs_d[NGS_IDX(d_cell_type[j], i)], 1);
        }
    }
    if (dist < 0.09) {  // inner disc for cell proliferation conditions
        // also increment per-type short-range flattened counters
        if (d_cell_type[j] >= 0 && d_cell_type[j] < N_TYPES) {
            atomicAdd(&d_ngs_90[NGS_IDX(d_cell_type[j], i)], 1);
        }
    }
    if (dist < 0.082) {  // 0.082
        if (d_cell_type[j] >= 0 && d_cell_type[j] < N_TYPES) {
            atomicAdd(&d_ngs_82[NGS_IDX(d_cell_type[j], i)], 1);
        }
    }
    if (dist < 0.075) {  // 0.075
        if (d_cell_type[j] >= 0 && d_cell_type[j] < N_TYPES) {
            atomicAdd(&d_ngs_75[NGS_IDX(d_cell_type[j], i)], 1);
        }
    }
    if (dist < 0.045) {  // 0.045
        if (d_cell_type[j] >= 0 && d_cell_type[j] < N_TYPES) {
            atomicAdd(&d_ngs_45[NGS_IDX(d_cell_type[j], i)], 1);
        }
    }

    if (dist > d_pm.r_max)  // dist = norm3df(r.x, r.y, r.z) solvers line 308
        return dF;          // cutoff for chemical and mechanical interaction

    if (d_cell_type[i] < 0 || d_cell_type[j] < 0) {
        return dF;  // cells in staging area have no interactions
    }

    if (i == j) {      // if the cell is interacting with itself
        dF += d_W[i];  // add stochasticity from the weiner process to the
                       // attributes of the cells
        return dF;
    }

    // Mechanical forces
    if (!d_pm.mov_switch)
        return dF;  // if cell movement is off, return no forces

    // default adhesion and repulsion vals for cell interactions
    float Rep = d_pm.Rxx * 0.2;  // because t=0.2*day
    float rep = d_pm.rxx;

    // const float interactions[5][5] = {
    //     // 1 means interact, 0 means no
    //     // rows: cell i type, cols: cell j type
    //     // M   Xd   Xl   Id   Il
    //     {1.0, 1.0, 0.0, 1.0, 0.0},  // M
    //     {0.0, 1.0, 1.0, 0.0, 0.0},  // Xd
    //     {0.0, 1.0, 1.0, 0.0, 0.0},  // Xl
    //     {0.0, 0.0, 0.0, 1.0, 1.0},  // Id
    //     {0.0, 0.0, 0.0, 1.0, 1.0}   // Il
    // };

    // const float interactions[5][5][2] = {
    //     // [i][j][0] = Rep, [i][j][1] = rep
    //     // rows: cell i type, cols: cell j type
    //     //      M             Xd             Xl           Id            Il
    //     {{0.02, 0.05}, {0.02, 0.067}, {0.00, 0.00}, {0.005, 0.03}, {0.00, 0.00}},  // M
    //     {{0.015, 0.067}, {0.016, 0.045}, {0.016, 0.045}, {0.00, 0.00}, {0.00, 0.00}},  // Xd
    //     {{0.00, 0.00}, {0.016, 0.045}, {0.016, 0.06}, {0.00, 0.00}, {0.00, 0.00}},  // Xl
    //     {{0.00, 0.00}, {-0.005, 0.09}, {0.00, 0.00}, {0.01, 0.04}, {0.035, 0.04}},  // Id
    //     {{0.00, 0.00}, {0.00, 0.00}, {0.00, 0.00}, {0.032, 0.04}, {0.032, 0.04}}   // Il
    // };
     const float interactions[5][5][2] = {
        // [i][j][0] = Rep, [i][j][1] = rep
        // rows: cell i type, cols: cell j type
        //      M             Xd             Xl           Id            Il
        {{0.02, 0.05}, {0.02, 0.067}, {0.00, 0.00}, {0.005, 0.03}, {0.00, 0.00}},  // M
        {{0.015, 0.067}, {0.016, 0.045}, {0.016, 0.045}, {0.00, 0.00}, {0.00, 0.00}},  // Xd
        {{0.00, 0.00}, {0.016, 0.045}, {0.016, 0.06}, {0.00, 0.00}, {0.00, 0.00}},  // Xl
        {{0.00, 0.00}, {0.00, 0.00}, {0.00, 0.00}, {0.01, 0.04}, {0.035, 0.04}},  // Id
        {{0.00, 0.00}, {0.00, 0.00}, {0.00, 0.00}, {0.032, 0.04}, {0.032, 0.04}}   // Il
    };

    if (d_pm.diff_adh_rep && d_cell_type[i] >= 0 && d_cell_type[j] >= 0) {
        Rep = interactions[d_cell_type[i]][d_cell_type[j]][0];  
        rep = interactions[d_cell_type[i]][d_cell_type[j]][1];        
    }


    // Volkening 2018
    float F = -Rep * (0.5f + 0.5f * tanhf((rep - dist) * 100));

    d_mech_str[i] -= F;  // mechanical strain is the sum of forces on the cell

    dF.x -= r.x * F / dist;
    dF.y -= r.y * F / dist;
    dF.z -= 0;

    // dF is the change in x,y,z,u,v etc. over dt, for a particular pairwise
    // interaction. Yalla sums the dFs for all interactions for cell i to give
    // d_dX[i] Yalla compute the new values by multiplying d_dX[i] by dt and
    // adding to the values in the current time step This function is in solvers
    // in the euler_step function

    return dF;
}

__global__ void generate_noise(int n, curandState* d_state)
{  // Weiner process for Heun's method
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // return noise for every attribute of the cell in this case x,y,z
    d_W[i].x =
        curand_normal(&d_state[i]) * powf(d_pm.dt, 0.5) * d_pm.noise / d_pm.dt;
    d_W[i].y =
        curand_normal(&d_state[i]) * powf(d_pm.dt, 0.5) * d_pm.noise / d_pm.dt;
    d_W[i].z = 0;
}


__global__ void stage_new_cells(int n_cells, curandState* d_state, float3* d_X,
    float3* d_old_v, int* d_n_cells, Mesh_d wall_mesh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // index of current cell
    if (i >= n_cells) return;
    if (n_cells >= (d_pm.n_max * 0.9)) return;
    float r1 = curand_uniform(&d_state[i]);
    float r2 = curand_uniform(&d_state[i]);

    if (i < d_pm.n_new_cells) {  // threads with i < n_new_cell create new cells
        int n = atomicAdd(d_n_cells, 1);
        d_X[n].x = d_tis_min.x + (d_tis_max.x - d_tis_min.x) * r1;
        d_X[n].y = d_tis_min.y + (d_tis_max.y - d_tis_min.y) * r2;

        d_old_v[n] = d_old_v[i];

        if (i < (d_pm.n_new_cells / 2)) {
            d_cell_type[n] = -2;  // M staging 
            d_X[n].z = 0;         
        } else {
            d_cell_type[n] = -3;  // X staging   
            d_X[n].z = 0;  
        }
        // Remove if outside mesh
        if (test_exclusion(wall_mesh, d_X[n])) {
            d_cell_type[n] = -99;  // Mark for removal
        }
    }
}

__global__ void proliferation(int n_cells, curandState* d_state, float3* d_X,
    float3* d_old_v, int* d_n_cells, Mesh_d wall_mesh)
{
    // change cells from staging to active types if conditions are met
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // index of current cell

    if (i >= n_cells) return;                   // stop if i >= n_cells
    if (n_cells >= (d_pm.n_max * 0.9)) return;  // no div above n_max

     if (d_cell_type[i] == 0){ // M
        if (d_ngs_d[NGS_IDX(1,i)] + d_ngs_d[NGS_IDX(3,i)] > 3 + 3.5f * d_ngs_d[NGS_IDX(0,i)] && 
            (d_ngs_82[NGS_IDX(1,i)] + d_ngs_82[NGS_IDX(0,i)] + d_ngs_82[NGS_IDX(3,i)]) <= 4)
         {
             int n = atomicAdd(d_n_cells, 1);
             float theta = curand_uniform(&d_state[i]) * 2 * M_PI;
             d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
             d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
             d_X[n].z = 0;
             d_old_v[n] = d_old_v[i];
             d_mech_str[n] = 0.0;
             d_cell_type[n] = 0;  
         }
     }
     if (d_cell_type[i] == -2) { // M staging
        if (d_ngs_82[NGS_IDX(0,i)] + d_ngs_82[NGS_IDX(1,i)] == 0){
            int n = atomicAdd(d_n_cells, 1);
             float theta = curand_uniform(&d_state[i]) * 2 * M_PI;
             d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
             d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
             d_X[n].z = 0;
             d_old_v[n] = d_old_v[i];
             d_mech_str[n] = 0.0;
             d_cell_type[n] = 0;  
        }
    }
    if (d_cell_type[i] == 1){ // Xd
        if (d_ngs_82[NGS_IDX(1,i)] + d_ngs_82[NGS_IDX(2,i)] <= 6) {
            int n = atomicAdd(d_n_cells, 1);
            float theta = curand_uniform(&d_state[i]) * 2 * M_PI;
            d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
            d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
            d_X[n].z = 0;
            d_old_v[n] = d_old_v[i];
            d_mech_str[n] = 0.0;
            d_cell_type[n] = 1;  
        }
    }
    if (d_cell_type[i] == 2){ //Xl
        if (d_ngs_82[NGS_IDX(1,i)] + d_ngs_82[NGS_IDX(2,i)] <= 1) {
            int n = atomicAdd(d_n_cells, 1);
            float theta = curand_uniform(&d_state[i]) * 2 * M_PI;
            d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
            d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
            d_X[n].z = 0;
            d_old_v[n] = d_old_v[i];
            d_mech_str[n] = 0.0;
            d_cell_type[n] = 2;  
        }    
    }
    if (d_cell_type[i] == -3) { // Xl staging
        if (d_ngs_82[NGS_IDX(1,i)] + d_ngs_82[NGS_IDX(2,i)] == 0){
            int n = atomicAdd(d_n_cells, 1);
            float theta = curand_uniform(&d_state[i]) * 2 * M_PI;
            d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
            d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
            d_X[n].z = 0;
            d_old_v[n] = d_old_v[i];
            d_mech_str[n] = 0.0;
            d_cell_type[n] = 2;  
        }
    }
    if (d_cell_type[i] == 3){ //Id
        if (d_ngs_82[NGS_IDX(3,i)] + d_ngs_82[NGS_IDX(4,i)] <= 7) {
            int n = atomicAdd(d_n_cells, 1);
            float theta = curand_uniform(&d_state[i]) * 2 * M_PI;
            d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
            d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
            d_X[n].z = 0;
            d_old_v[n] = d_old_v[i];
            d_mech_str[n] = 0.0;
            d_cell_type[n] = 3;  
        }
    }
    if (d_cell_type[i] == 4){ //Il
        if (d_ngs_82[NGS_IDX(3,i)] + d_ngs_82[NGS_IDX(4,i)] <= 7) {
            int n = atomicAdd(d_n_cells, 1);
            float theta = curand_uniform(&d_state[i]) * 2 * M_PI;
            d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
            d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
            d_X[n].z = 0;
            d_old_v[n] = d_old_v[i];
            d_mech_str[n] = 0.0;
            d_cell_type[n] = 4;  
        }
    }
}

__global__ void death(
    int n_cells, curandState* d_state, float3* d_X, int* d_n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    float r = curand_uniform(&d_state[i]);
    // if (d_cell_type[i] == -1 || d_cell_type[i] == -2) return;  // don't die

    if (d_cell_type[i] == 0) { // M
        if ((d_ngs_90[NGS_IDX(1,i)] > 1.25f * d_ngs_90[NGS_IDX(0,i)]) ||
            (d_ngs_d[NGS_IDX(0,i)] >= 2 * d_ngs_d[NGS_IDX(1,i)] &&
             d_ngs_45[NGS_IDX(4,i)] < 3 &&
            curand_uniform(&d_state[i]) < 0.0333f)) {
             d_cell_type[i] = -1;
         }
     }
}

__global__ void cell_switching(
    int n_cells, curandState* d_state, float3* d_X, const float* d_slow_reg, 
    Plane* d_ray_plane)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    if (d_cell_type[i] == 1) { // Xd
        if (d_ngs_75[NGS_IDX(4,i)] > 2 + d_ngs_45[NGS_IDX(3,i)]) {
             d_cell_type[i] = 2; // Xd -> Xl
         }
     }
    if (d_cell_type[i] == 2) { // Xl
        if (d_ngs_45[NGS_IDX(3,i)] + 
            ((curand_uniform(&d_state[i]) > 0.5) * d_ngs_75[NGS_IDX(1,i)]) > // Bernoulli
            1 + d_ngs_45[NGS_IDX(4,i)] + d_ngs_75[NGS_IDX(0,i)]) {
             d_cell_type[i] = 1; // Xl -> Xd
         }
     }
    if (d_cell_type[i] == 3) { // Id
        if (d_ngs_90[NGS_IDX(0,i)] > 3 ||
            (d_ngs_d[NGS_IDX(1,i)] > 5 && d_ngs_75[NGS_IDX(1,i)] < 2)) {
                d_cell_type[i] = 4; // Id -> Il
            }
     }
    if (d_cell_type[i] == 4) { // Il
        if ((d_ngs_90[NGS_IDX(0,i)] < 3 && d_ngs_d[NGS_IDX(1,i)] < 9) ||
            (d_ngs_90[NGS_IDX(0,i)] < 3 && d_ngs_75[NGS_IDX(1,i)] > 3)) {
                d_cell_type[i] = 3; // Il -> Id
            }
     }
}


template<typename MeshType>
void update_slow_reg(MeshType& tis, float slow_reg[2])
{
    // host function for updating slow advection region
    // slow region is a horizontal band across the tissue
    float p_min = tis.get_minimum().y;
    float p_max = tis.get_maximum().y;

    float center = (p_max + p_min) / 2;
    float band_width =
        (h_pm.s_slow * (p_max - p_min));   // fraction of total size
    float p1 = center + (band_width / 2);  // p1 is top of band
    float p2 = center - (band_width / 2);
    slow_reg[0] = p1;
    slow_reg[1] = p2;
    // cudaMemcpy(d_slow_reg, &slow_reg, 2 * sizeof(float),
    //     cudaMemcpyHostToDevice);  // copy to device
    // std::cout << "slow region y: " << p1 << " to " << p2 << "\n";
}

__global__ void advection(int n_cells, const float3* d_X, float3* d_dX,
    const float* d_slow_reg, Plane* d_ray_plane, int time_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    // printf("slow_reg: %f, %f\n", (*slow_reg)[0], (*slow_reg)[1]);

    if ((d_X[i].y < d_slow_reg[0]) && (d_X[i].y > d_slow_reg[1]) &&
        (signed_distance_to_plane(d_X[i], *d_ray_plane) < 0)) {
        d_in_slow[i] = true;  // mark cells as in slow region
    } else {
        d_in_slow[i] = false;
    }

    if (d_cell_type[i] == 1) {  // only type 1 cells advect
        float norm_x = (d_X[i].x - d_tis_min.x) /
                       (d_tis_max.x - d_tis_min.x);  // frac along x axis
        float ad_time =  // wait time proportional to x frac, total duration
            norm_x * d_pm.cont_time * d_pm.ad_mult;  // and wait multiplier

        float ad = d_pm.ad_s;  // default advection strength
        if (d_pm.ad_func == 1 && time_step < ad_time) ad = 0;

        if (d_pm.slow_switch && d_in_slow[i]) {
            ad = d_pm.soft_ad_s;  // soft_ad if in slow_region
        }
        d_dX[i].x += ad * cosf(d_pm.ad_ang * M_PI / 180.0);  // angle in degrees
        d_dX[i].y -= ad * sinf(d_pm.ad_ang * M_PI / 180.0);  // angle in degrees
    }
}

__global__ void wall_forces_new(
    int n_cells, const float3* d_X, float3* d_dX, Mesh_d wall_mesh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    if (d_cell_type[i] < 0) return;  // no wall forces for staging cells

    float min_dist2 = 1e30f;
    float3 closest_nm;
    for (int j = 0; j < wall_mesh.n_facets; ++j) {
        float3 pt = closest_point_on_triangle(d_X[i], wall_mesh.d_facets[j]);
        float dx = d_X[i].x - pt.x;
        float dy = d_X[i].y - pt.y;
        float dz = d_X[i].z - pt.z;
        float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            closest_nm = wall_mesh.d_facets[j].n;
            // Optionally, store the triangle normal as well
        }
    }

    // // Determine if cell outside fin using ray-casting
    // bool outside = test_exclusion(wall_mesh, d_X[i]);
    // if (outside) {
    //     auto F_mag = fmaxf(-min_dist2, 0);  // force magnitude
    //     d_dX[i].x +=
    //         closest_nm.x * F_mag;  // force is product of displ and norm vec
    //     d_dX[i].y += closest_nm.y * F_mag;
    // }
    bool outside = test_exclusion(wall_mesh, d_X[i]);
    if (outside) {
        auto F_mag = fminf(sqrtf(min_dist2) * 20, 1.0f);  // force magnitude
        d_dX[i].x +=
            closest_nm.x * F_mag;  // force is product of displ and norm vec
        d_dX[i].y += closest_nm.y * F_mag;
    }
}

template<typename Pt>
__global__ void grow_cells(int n_cells, Pt* d_X, float stretchfactor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    d_X[i].x *= stretchfactor;
    d_X[i].y *= stretchfactor;
    // z remains unchanged
}

// Predicate for dead cells
struct is_dead {
    __host__ __device__ bool operator()(const thrust::tuple<int>& t) const
    {
        return thrust::get<0>(t) < 0;  // cell_type < 0 means dead
    }
};

void compact_cells_with_remove_if(int n_cells, float3* d_X, float3* d_W,
    int* d_cell_type, float* d_mech_str, bool* d_in_slow, int* d_ngs_45, 
    int* d_ngs_75, int* d_ngs_82, int* d_ngs_90, int* d_ngs_d, int* d_n_cells)
{
    // Create zip iterators for all arrays
    auto first = thrust::make_zip_iterator(thrust::make_tuple(d_cell_type, d_X,
        d_W, d_mech_str, d_in_slow, d_ngs_45, d_ngs_75, d_ngs_82, d_ngs_90, d_ngs_d));
    auto last = first + n_cells;

    // Remove dead cells (cell_type < 0) from all arrays
    auto new_end = thrust::remove_if(thrust::device, first, last,
        thrust::make_zip_iterator(thrust::make_tuple(d_cell_type)), is_dead());

    // Update n_cells
    int new_n = new_end - first;
    cudaMemcpy(d_n_cells, &new_n, sizeof(int), cudaMemcpyHostToDevice);
}


//////////////////////////////////////////////////////////////////////
// Main simulation function
//////////////////////////////////////////////////////////////////////

int tissue_sim(int argc, char const* argv[], int walk_id = 0, int step = 0,
    bool last_vtk_only = true, std::string out_dir_name = "output")
{
    std::cout << std::fixed
              << std::setprecision(6);  // set precision for floats
    // h_pm.dt = 0.05 * 0.6 * 0.6 / h_pm.D_v;

    // Prepare Random Variable for the Implementation of the Wiener Process
    curandState* d_state;  // define the random number generator on the GPu
    cudaMalloc(&d_state,
        h_pm.n_max * sizeof(curandState));  // allocate GPU memory according
                                            // to no. cells
    auto seed =
        time(NULL);  // random number seed - coupled to the time on your machine
    setup_rand_states<<<(h_pm.n_max + 32 - 1) / 32, 32>>>(h_pm.n_max, seed,
        d_state);  // configuring the random number generator
                   // on the GPU (provided by utils.cuh)
    std::mt19937 h_state(std::random_device{}()); // host random number generator


    /* create host variables*/
    // you first create an instance of the Property class on the host, then
    // you connect it to the global variable defined on the device with
    Property<float3> W{
        h_pm.n_max, "wiener_process"};  // weiner process random number
    cudaMemcpyToSymbol(d_W, &W.d_prop, sizeof(d_W));
    Property<float> mech_str{h_pm.n_max, "mech_str"};
    cudaMemcpyToSymbol(d_mech_str, &mech_str.d_prop, sizeof(d_mech_str));
    Property<int> cell_type{h_pm.n_max, "cell_type"};  // cell type labels
    cudaMemcpyToSymbol(d_cell_type, &cell_type.d_prop, sizeof(d_cell_type));
    Property<bool> in_slow{
        h_pm.n_max, "in_slow"};  // whether cell in slow region or not
    cudaMemcpyToSymbol(d_in_slow, &in_slow.d_prop, sizeof(d_in_slow));
    cudaMemcpyToSymbol(d_pm, &h_pm, sizeof(Pm));  // copy host params

    // flattened per-type neighbour counters (types 0..4)
    Property<int> ngs_45{h_pm.n_max * N_TYPES, "ngs"};
    cudaMemcpyToSymbol(d_ngs_45, &ngs_45.d_prop, sizeof(d_ngs_45));
    Property<int> ngs_75{h_pm.n_max * N_TYPES, "ngs"};
    cudaMemcpyToSymbol(d_ngs_75, &ngs_75.d_prop, sizeof(d_ngs_75));
    Property<int> ngs_82{h_pm.n_max * N_TYPES, "ngs"};
    cudaMemcpyToSymbol(d_ngs_82, &ngs_82.d_prop, sizeof(d_ngs_82));
    Property<int> ngs_90{h_pm.n_max * N_TYPES, "ngs"};
    cudaMemcpyToSymbol(d_ngs_90, &ngs_90.d_prop, sizeof(d_ngs_90));
    Property<int> ngs_d{h_pm.n_max * N_TYPES, "ngs_d"};
    cudaMemcpyToSymbol(d_ngs_d, &ngs_d.d_prop, sizeof(d_ngs_d));
    // zero host-side storage
    for (int k = 0; k < h_pm.n_max * N_TYPES; ++k) {
        ngs_45.h_prop[k] = 0;
        ngs_75.h_prop[k] = 0;
        ngs_82.h_prop[k] = 0;
        ngs_90.h_prop[k] = 0;
        ngs_d.h_prop[k] = 0;
    }

    // Initial conditions
    // Solution<Cell, Gabriel_solver> cells{h_pm.n_max, h_pm.g_size,
    // h_pm.r_max};
    Solution<float3, Grid_solver> cells{h_pm.n_max, h_pm.g_size, h_pm.c_size};
    // args are n_max, grid_size, cube_size
    float slow_reg[2] = {0, 0};  // initialise slow_reg with default values
    float* d_slow_reg;
    cudaMalloc(&d_slow_reg, 2 * sizeof(float));  // GPU mem alloc
    Plane ray_plane;
    Plane* d_ray_plane;
    cudaMalloc(&d_ray_plane, sizeof(Plane));  // GPU mem alloc
    Fin fin("../inits/fin_init.vtk",
        "fin_" + std::to_string(walk_id) + "_" + std::to_string(step));
    Fin fin_rays("../inits/ray_init.vtk",
        "fin_rays_" + std::to_string(walk_id) + "_" + std::to_string(step));


    if (h_pm.tmode == 0) {
        random_disk_z(h_pm.init_dist, cells);
        for (int i = 0; i < h_pm.n_0; i++) {
            cell_type.h_prop[i] = (std::rand() % 100 < h_pm.A_init)
                                      ? 1
                                      : 2;  // randomly assign a proportion of
                                            // initial cells with each type
        }
    }
    if (h_pm.tmode == 1) {
        regular_rectangle(h_pm.init_dist,
            std::round(std::sqrt(h_pm.n_0) / 10) * 10,
            cells);  // initialise rectangle specifying the no. cells along
                     // the x axis
        for (int i = 0; i < h_pm.n_0; i++) {
            cell_type.h_prop[i] = (std::rand() % 100 < h_pm.A_init) ? 1 : 2;
        }
    }
    if (h_pm.tmode == 2) {  // rectangle with spots on one end
        auto sp_size =
            (h_pm.A_init / 100.0) * h_pm.n_0;  // calculate no. cells in spot
        regular_rectangle_w_spot(sp_size, h_pm.init_dist,
            std::round(std::sqrt(h_pm.n_0) / 10) * 10, cells);
        for (int i = 0; i < h_pm.n_0; i++) {
            cell_type.h_prop[i] = (i < h_pm.n_0 - sp_size)
                                      ? 2
                                      : 1;  // set cell type to 1 for spot
                                            // cells, and 2 for all others
        }
    }
    if (h_pm.tmode ==
        3) {  // cut the tissue mesh out of a random cloud of cells
        Mesh tis{"../inits/fin_init.vtk"};
        tis.rescale(h_pm.tis_s);  // expand the mesh to fit to the boundaries
        auto tis_min = tis.get_minimum();
        auto tis_max = tis.get_maximum();
        cudaMemcpyToSymbol(d_tis_min, &tis_min, sizeof(float3));  // tis min
        cudaMemcpyToSymbol(d_tis_max, &tis_max, sizeof(float3));  // tis max
        random_rectangle(
            h_pm.init_dist, tis.get_minimum(), tis.get_maximum(), cells);
        auto new_n =
            thrust::remove_if(thrust::host, cells.h_X, cells.h_X + *cells.h_n,
                [&tis](float3 x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        std::uniform_int_distribution<int> dist(0, N_TYPES - 1); // for random cell type assignment
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            // cell_type.h_prop[i] = (std::rand() % 100 < h_pm.A_init)
            //                           ? 1   // set cell type to 1 for spot
            //                           : 2;  // cells, and 2 for all others
            cell_type.h_prop[i] = dist(h_state);
        }
        update_slow_reg(tis, slow_reg);
        cudaMemcpy(
            d_slow_reg, slow_reg, 2 * sizeof(float), cudaMemcpyHostToDevice);
        ray_plane = fin_rays.get_3rd_ray_plane();
        cudaMemcpy(
            d_ray_plane, &ray_plane, sizeof(Plane), cudaMemcpyHostToDevice);
    }
    if (h_pm.tmode == 4) {  // cut the fin mesh out of a random cloud of cells
        Mesh tis{"../inits/fin_init.vtk"};
        tis.rescale(h_pm.tis_s);
        auto tis_min = tis.get_minimum();
        auto tis_max = tis.get_maximum();
        cudaMemcpyToSymbol(d_tis_min, &tis_min, sizeof(float3));  // tis min
        cudaMemcpyToSymbol(d_tis_max, &tis_max, sizeof(float3));  // tis max
        auto x_len = tis.get_maximum().x - tis.get_minimum().x;
        random_rectangle(
            h_pm.init_dist, tis.get_minimum(), tis.get_maximum(), cells);
        auto new_n =
            thrust::remove_if(thrust::host, cells.h_X, cells.h_X + *cells.h_n,
                [&tis](float3 x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            // spot cells appear in leftmost 10% of tissue
            if (cells.h_X[i].x < tis.get_minimum().x + (x_len * 0.1))
                cell_type.h_prop[i] = (std::rand() % 100 < 50) ? 1 : 2;
            else
                cell_type.h_prop[i] = 2;
        }
        update_slow_reg(tis, slow_reg);
        cudaMemcpy(
            d_slow_reg, slow_reg, 2 * sizeof(float), cudaMemcpyHostToDevice);
        ray_plane = fin_rays.get_3rd_ray_plane();
        cudaMemcpy(
            d_ray_plane, &ray_plane, sizeof(Plane), cudaMemcpyHostToDevice);
    }
    if (h_pm.tmode == 5) {  // fin with spot aggregation at top
        Mesh tis{"../inits/fin_init.vtk"};
        // Mesh tis{"../data/lmk_DA-1-10_12-09-25/DA-1-10_12-07_0_lmk.vtk"};
        tis.rescale(h_pm.tis_s);
        auto tis_min = tis.get_minimum();
        auto tis_max = tis.get_maximum();
        cudaMemcpyToSymbol(d_tis_min, &tis_min, sizeof(float3));  // tis min
        cudaMemcpyToSymbol(d_tis_max, &tis_max, sizeof(float3));  // tis max
        auto y_len = tis.get_maximum().y - tis.get_minimum().y;
        random_rectangle(
            h_pm.init_dist, tis.get_minimum(), tis.get_maximum(), cells);
        auto new_n =
            thrust::remove_if(thrust::host, cells.h_X, cells.h_X + *cells.h_n,
                [&tis](float3 x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            // spot cells appear in topmost 15% of tissue
            if (cells.h_X[i].y > tis.get_maximum().y - (y_len * 0.15)) {
                if (std::rand() % 100 < 50) {
                    cell_type.h_prop[i] = 4;
                    // cells.h_X[i].z = 0;  // I layer z=0
                } else {
                    cell_type.h_prop[i] = 2;
                    // cells.h_X[i].z = 0 + h_pm.s_layer;  // X above I
                }
            } else {
                cell_type.h_prop[i] = 2;
                // cells.h_X[i].z = 0 + h_pm.s_layer;  // X above I
            }
        };
        update_slow_reg(tis, slow_reg);
        cudaMemcpy(
            d_slow_reg, slow_reg, 2 * sizeof(float), cudaMemcpyHostToDevice);
        ray_plane = fin_rays.get_3rd_ray_plane();
        cudaMemcpy(
            d_ray_plane, &ray_plane, sizeof(Plane), cudaMemcpyHostToDevice);
    }

    // Initialise properties and k with zeroes
    for (int i = 0; i < h_pm.n_max; i++) {
        mech_str.h_prop[i] = 0;
        in_slow.h_prop[i] = false;
        // zero flattened per-type arrays
        for (int t = 0; t < N_TYPES; ++t) {
            int idx = t * h_pm.n_max + i;
            ngs_45.h_prop[idx] = 0;
            ngs_75.h_prop[idx] = 0;
            ngs_82.h_prop[idx] = 0;
            ngs_90.h_prop[idx] = 0;
            ngs_d.h_prop[idx] = 0;
        }
    }

    int time_step;  // declare  outside main loop for access in gen_func
    auto generic_function = [&](const int n, const float3* __restrict__ d_X,
                                float3* d_dX) {
        // Set these properties to zero after every timestep so they
        // don't accumulate Called every timesetep, allows you to add
        // custom forces at every timestep e.g. advection
        thrust::fill(thrust::device, mech_str.d_prop,
            mech_str.d_prop + cells.get_d_n(), 0.0);
        thrust::fill(thrust::device, in_slow.d_prop,
            in_slow.d_prop + cells.get_d_n(), false);
        // zero flattened per-type arrays on device (cells.get_d_n() * N_TYPES)
        thrust::fill(thrust::device, ngs_45.d_prop,
            ngs_45.d_prop + (cells.get_d_n() * N_TYPES), 0);
        thrust::fill(thrust::device, ngs_75.d_prop,
            ngs_75.d_prop + (cells.get_d_n() * N_TYPES), 0);
        thrust::fill(thrust::device, ngs_82.d_prop,
            ngs_82.d_prop + (cells.get_d_n() * N_TYPES), 0);
        thrust::fill(thrust::device, ngs_90.d_prop,
            ngs_90.d_prop + (cells.get_d_n() * N_TYPES), 0);
        thrust::fill(thrust::device, ngs_d.d_prop,
            ngs_d.d_prop + (cells.get_d_n() * N_TYPES), 0);

        // return wall_forces<Cell, boundary_force>(n, d_X, d_dX, 0);
        if (h_pm.adv_switch)
            advection<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_X, d_dX, d_slow_reg, d_ray_plane, time_step);
        if (h_pm.fin_walls)
            wall_forces_new<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_X, d_dX, fin.mesh);
    };

    cells.copy_to_device();
    mech_str.copy_to_device();
    cell_type.copy_to_device();
    in_slow.copy_to_device();
    ngs_45.copy_to_device();
    ngs_75.copy_to_device();
    ngs_82.copy_to_device();
    ngs_90.copy_to_device();
    ngs_d.copy_to_device();

    Vtk_output output{
        "out_" + std::to_string(walk_id) + "_" + std::to_string(step),
        out_dir_name};

    // initialise properties by taking a zero dt step
    cells.take_step<pairwise_force>(0.0, generic_function);

    if (!last_vtk_only) {
        // write out initial condition
        cells.copy_to_host();
        mech_str.copy_to_host();
        cell_type.copy_to_host();
        in_slow.copy_to_host();
        ngs_45.copy_to_host();
        ngs_75.copy_to_host();
        ngs_82.copy_to_host();
        ngs_90.copy_to_host();
        ngs_d.copy_to_host();

        output.write_positions(cells);
        output.write_property(mech_str);
        output.write_property(cell_type);
        output.write_property(in_slow);
        // int n_cells = *cells.h_n;
        // const char* ngs_names[N_TYPES] = {"ngs_M", "ngs_Xd", "ngs_Xl",
        //                                     "ngs_Id", "ngs_Il"};
        // for (int t = 0; t < N_TYPES; ++t) {
        //     Property<int> ngs_t{h_pm.n_max, ngs_names[t]};
        //     for (int i = 0; i < n_cells; ++i) {
        //         int flat_idx = t * h_pm.n_max + i;
        //         ngs_t.h_prop[i] = ngs.h_prop[flat_idx];
        //     }
        //     cudaMemcpy(ngs_t.d_prop, ngs_t.h_prop,
        //                 n_cells * sizeof(int), cudaMemcpyHostToDevice);
        //     output.write_property(ngs_t);
        // }
        fin.write_vtk();
        fin_rays.write_vtk();
    }

    // Main simulation loop
    for (time_step = 0; time_step <= h_pm.cont_time; time_step++) {
        if (h_pm.t_grow_switch && time_step > 0 && time_step % 10 == 0) {
            float stretchfactor = fin.grow(
                h_pm.t_growth_rate * 0.2 * 10);  // grow the fin by 10% every
            float3 tis_min = fin.get_minimum();
            float3 tis_max = fin.get_maximum();
            cudaMemcpyToSymbol(d_tis_min, &tis_min, sizeof(float3));
            cudaMemcpyToSymbol(d_tis_max, &tis_max, sizeof(float3));
            // 10 timesteps
            grow_cells<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), cells.d_X, stretchfactor);
            update_slow_reg(fin, slow_reg);
            cudaMemcpy(d_slow_reg, &slow_reg, 2 * sizeof(float),
                cudaMemcpyHostToDevice);  // copy to device
            fin_rays.grow(h_pm.t_growth_rate * 0.2 * 10);
            ray_plane = fin_rays.get_3rd_ray_plane();
            cudaMemcpy(d_ray_plane, &ray_plane, sizeof(Plane),
                cudaMemcpyHostToDevice);  // copy to device
        }
        for (float T = 0.0; T < 1.0; T += h_pm.dt) {
            // printf("T = %f\n", T);
            generate_noise<<<(cells.get_d_n() + 32 - 1) / 32, 32>>>(
                cells.get_d_n(),
                d_state);  // generate random noise which we will use later
                           // on to move the cells
            if (h_pm.prolif_switch) {
                if (time_step % int(h_pm.cont_time / 500) == 0) {
                    stage_new_cells<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                        cells.get_d_n(), d_state, cells.d_X, cells.d_old_v,
                        cells.d_n, fin.mesh);  // stage new cells
                }
                cells.take_step<pairwise_force>(0.0, generic_function);
                proliferation<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), d_state, cells.d_X, cells.d_old_v,
                    cells.d_n, fin.mesh);  // simulate proliferation
            }
            if (h_pm.type_switch)
                cell_switching<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), d_state, cells.d_X, d_slow_reg,
                    d_ray_plane);   // switch cell types if
                                    // conditions are met
            if (h_pm.death_switch)  // death occurs once per day - 20 days total
                                    // if (time_step % int(h_pm.cont_time / 20)
                                    // == 0) {
                death<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), d_state, cells.d_X, cells.d_n);
            // }
            compact_cells_with_remove_if(cells.get_d_n(), cells.d_X, W.d_prop,
                cell_type.d_prop, mech_str.d_prop, in_slow.d_prop, ngs_45.d_prop,
                ngs_75.d_prop, ngs_82.d_prop, ngs_90.d_prop, ngs_d.d_prop, cells.d_n);

            cells.take_step<pairwise_force, friction_on_background>(
                h_pm.dt, generic_function);
        }

        if (!last_vtk_only &&
                time_step % int(h_pm.cont_time / h_pm.no_frames) == 0 ||
            (time_step == h_pm.cont_time)) {
            cells.copy_to_host();
            mech_str.copy_to_host();
            cell_type.copy_to_host();
            in_slow.copy_to_host();
            ngs_45.copy_to_host();
            ngs_75.copy_to_host();
            ngs_82.copy_to_host();
            ngs_90.copy_to_host();
            ngs_d.copy_to_host();

            output.write_positions(cells);
            output.write_property(mech_str);
            output.write_property(cell_type);
            output.write_property(in_slow);
            // int n_cells = *cells.h_n;
            // const char* ngs_names[N_TYPES] = {"ngs_M", "ngs_Xd", "ngs_Xl",
            //                                   "ngs_Id", "ngs_Il"};
            // for (int t = 0; t < N_TYPES; ++t) {
            //     Property<int> ngs_t{h_pm.n_max, ngs_names[t]};
            //     for (int i = 0; i < n_cells; ++i) {
            //         int flat_idx = t * h_pm.n_max + i;
            //         ngs_t.h_prop[i] = ngs.h_prop[flat_idx];
            //     }
            //     cudaMemcpy(ngs_t.d_prop, ngs_t.h_prop,
            //                n_cells * sizeof(int), cudaMemcpyHostToDevice);
            //     output.write_property(ngs_t);
            // }
            fin.write_vtk();
            fin_rays.write_vtk();
        }
    }
    return 0;
}

// compile tissue_sim as main when this file is not included as library
// elsewhere
#ifndef COMPILE_AS_LIBRARY
int main(int argc, char const* argv[])
{
    bool last_vtk_only = false;
    std::string out_dir_name = "output";
    int walk_id = 0, step = 0;
    if (argc > 1) walk_id = std::atoi(argv[1]);
    if (argc > 2) step = std::atoi(argv[2]);
    if (argc > 3) out_dir_name = std::string(argv[3]);
    return tissue_sim(argc, argv, walk_id, step, last_vtk_only, out_dir_name);
}
#endif