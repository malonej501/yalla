// Toy model for accessing cell fate decisions during cancer development

// Compilation
//
// $ nvcc -std=c++14 -arch=sm_86 {"compiler flags"} Limb_model_simulation.cu
// The values for "-std" and "-arch" flags will depend on your version of CUDA
// and the specific GPU model you have respectively. e.g. -std=c++14 works for
// CUDA version 11.6 and -arch=sm_86 corresponds to the generation of NVIDIA
// Geforce 30XX cards.
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <cmath>
#include <filesystem>
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
MAKE_PT(Cell, u, v);

// define global variables for the GPU
__device__ float* d_mech_str;
__device__ int* d_cell_type;  // cell_type: A=1, B=2, DEAD=0
__device__ Cell* d_W;  // random number from Weiner process for stochasticity
__device__ bool* d_in_ray;    // whether a cell is in a ray
__device__ Pm d_pm;           // simulation parameters (host h_pm)
__device__ float3 d_tis_min;  // min coordinate of tissue mesh
__device__ float3 d_tis_max;  // max coordinate of tissue mesh

template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    if (dist > d_pm.r_max)  // dist = norm3df(r.x, r.y, r.z) solvers line 308
        return dF;          // set cutoff for computing interaction forces

    if (i == j) {      // if the cell is interacting with itself
        dF += d_W[i];  // add stochasticity from the weiner process to the
                       // attributes of the cells

        // Chemical production and degredation

        if (d_pm.cmode == 0) {  // chemical production and degredation
            dF.u = d_pm.k_prod * (1.0 - Xi.u) *
                   (d_cell_type[i] == 1 ||
                       d_cell_type[i] == 3);  // cell type 1/3 produce u
            dF.v = d_pm.k_prod * (1.0 - Xi.v) *
                   (d_cell_type[i] == 2);  // cell type 2 produces chemical v
            // dF.u = d_pm.k_prod * ((d_cell_type[i] == 1 || d_cell_type[i] ==
            // 3) &
            //                          Xi.u < 1);  // stop making u when it
            //                                      //   reaches 1
            // dF.v = d_pm.k_prod *
            //        ((d_cell_type[i] == 2) & Xi.v < 1);  // stop making v when
            // it reaches 1
            dF.u -= d_pm.k_deg * (Xi.u);
            dF.v -= d_pm.k_deg * (Xi.v);
        }

        if (d_pm.cmode == 1) {
            // see Schnackenberg 1979 eq. 41
            float a = ((Xi.x + 3) * 0.1);
            float b = ((Xi.y + 1) * 0.2);
            // dF.u = (Xi.u * Xi.u * Xi.v) - Xi.u + d_pm.a_u;
            // dF.v = -(Xi.u * Xi.u * Xi.v) + d_pm.b_v;
            dF.u = (Xi.u * Xi.u * Xi.v) - Xi.u + a;
            dF.v = b - (Xi.u * Xi.u * Xi.v);
            // dF.u = (Xi.u * Xi.u * Xi.v) - Xi.u + (Xi.x * 0.1);
            // dF.v = -(Xi.u * Xi.u * Xi.v) + (Xi.y * 0.1);
        }
        if (d_pm.cmode == 2) {
            // Gray Scott model
            float a = 0.3;
            float b = 0.003;
            float R = 0.1;
            // float a = ((Xi.x + 3) * 0.1);
            // float b = ((Xi.y + 1) * 0.01);
            // float R = ((Xi.x + 3) * 0.1);
            dF.u = R * ((Xi.u * Xi.u * Xi.v) - ((a + b) * Xi.u));
            dF.v = R * (-(Xi.u * Xi.u * Xi.v) + (a * (1 - Xi.v)));
        }
        if (d_pm.cmode == 3) {
            // Gierer Meinhardt model
            // float a = 0.8;
            // float b = 1;
            // float c = 6;
            // float a = ((Xi.x + 3) * 0.1);
            // float b = ((Xi.y + 1) * 0.1);
            // float c = ((Xi.y + 1) * 0.1);
            // dF.u = (a + ((Xi.u * Xi.u) / Xi.v) - (b * Xi.u));
            // dF.v = (Xi.u * Xi.u) - (c * Xi.v);
            const auto lambda = 1;
            const auto f_v = 0.1;
            const auto f_u = 10.0;
            const auto g_u = 5.0;
            const auto m_u = 0.02;
            const auto m_v = 0.05;
            const auto s_u = 0.005;
            dF.u = lambda *
                   ((f_u * Xi.u * Xi.u) / (1 + f_v * Xi.v) - m_u * Xi.u + s_u);
            dF.v = lambda * (g_u * Xi.u * Xi.u - m_v * Xi.v);
        }
        return dF;
    }

    // Diffusion
    dF.u = -d_pm.D_u * r.u;  // r = Xi - Xj solvers.cuh line 448
    dF.v = -d_pm.D_v * r.v;
    // dF.u = -((Xi.x + 3) * 0.01) * r.u;
    // dF.v = -((Xi.y + 1.5) * 0.01) * r.v;
    // dF.u = -0.1 * r.u;
    // dF.v = -4 * r.v;
    // dF.u = -Xi.x * r.u * 0.01;
    // dF.v = -Xi.y * r.v * 0.01;
    // dF.u = -1 * r.u;
    // dF.v = -40 * r.v;
    // dF.u = -0.01 * r.u;
    // dF.v = -0.05 * r.v;


    // Mechanical forces

    if (!d_pm.mov_switch)
        return dF;  // if cell movement is off, return no forces

    // default adhesion and repulsion vals for cell interactions
    float Adh = d_pm.Add;
    float adh = d_pm.add;
    float Rep = d_pm.Rdd;
    float rep = d_pm.rdd;

    if (d_pm.diff_adh_rep) {
        if ((d_cell_type[i] == 1 and d_cell_type[j] == 1) or
            (d_cell_type[i] == 3 and d_cell_type[j] == 3) or
            (d_cell_type[i] == 1 and d_cell_type[j] == 3) or
            (d_cell_type[i] == 3 and d_cell_type[j] == 1)) {
            Adh = d_pm.Aii;  // A-A interact with different adh and rep vals
            adh = d_pm.aii;
            Rep = d_pm.Rii;
            rep = d_pm.rii;
        }
    }


    // float F = (k_rep * fmaxf(0.08 - dist, 0) - k_adh * fmaxf(dist - 0.08,
    // 0)); // forces are also dependent on adhesion and repulsion between cell
    // types float F = (Adh * r.x * exp(-sqrt(r.x^2 + r.y^2) / adh)) / (adh *
    // sqrt(r.x^2 + r.y^2)) - (Rep * r.x * exp(-sqrt(r.x^2 - r.y^2) / rep) /
    // (rep * sqrt(r.x^2 - r.y^2))); Volkening et al. 2015 force potential,
    // function in terms of distance in n dimensions
    float term1 = Adh / adh * expf(-dist / adh);
    float term2 = Rep / rep * expf(-dist / rep);
    float F = term1 - term2;
    // printf("%f\n", F);
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
    d_W[i].u = 0;
    d_W[i].v = 0;
}

__global__ void proliferation(int n_cells, curandState* d_state, Cell* d_X,
    float3* d_old_v, int* d_n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // index of current cell
    if (i >= n_cells) return;                       // stop if i >= n_cells
    if (n_cells >= (d_pm.n_max * 0.9)) return;      // no div above n_max
    if (d_mech_str[i] > d_pm.mech_thresh) return;   // no div above mech_thresh

    if (d_cell_type[i] == 1) {
        // if (d_X[i].u > 0.85) return;
        if (curand_uniform(&d_state[i]) > (d_pm.A_div * d_pm.dt)) return;
    }

    if (d_cell_type[i] == 2) {
        if (curand_uniform(&d_state[i]) > (d_pm.B_div * d_pm.dt)) return;
    }

    if (d_cell_type[i] == 3) {  // type 3 cells same as type 1
        // if (d_X[i].v < d_pm.vthresh) return;  // divide if in v spot
        if (curand_uniform(&d_state[i]) > (d_pm.A_div * d_pm.dt)) return;
    }

    int n = atomicAdd(d_n_cells, 1);

    float theta = curand_uniform(&d_state[i]) * 2 * M_PI;  // child pos -2D only
    d_X[n].x = d_X[i].x + (d_pm.div_dist * cosf(theta));
    d_X[n].y = d_X[i].y + (d_pm.div_dist * sinf(theta));
    d_X[n].z = 0;

    d_old_v[n] = d_old_v[i];

    d_mech_str[n] = 0.0;


    // set child cell types
    if (d_cell_type[i] == 2) {  // type 2 cells (non-spot) can divide into type
                                // 1 cells when d_pmode is set correctly
        if (d_pm.pmode == 0) d_cell_type[n] = 2;
        if (d_pm.pmode == 1)
            d_cell_type[n] =
                (curand_uniform(&d_state[i]) < d_pm.r_A_birth)
                    ? 1
                    : 2;  // sometimes cell type 2 produces cell type 1 random
                          // birth of cell type 1 is inhibited by chemical u
        if (d_pm.pmode == 2)
            d_cell_type[n] = (curand_uniform(&d_state[i]) < d_pm.r_A_birth &&
                                 d_X[i].u < d_pm.uthresh)
                                 ? 1
                                 : 2;
    }
    if (d_cell_type[i] == 1) d_cell_type[n] = 1;
    if (d_cell_type[i] == 3) d_cell_type[n] = 3;

    // half the amount of each chemical upon cell division in the parent cell
    d_X[i].u *= 0.5;
    d_X[i].v *= 0.5;
    // the child inherits the other half of the amount of the chemical
    d_X[n].u = d_X[i].u;
    d_X[n].v = d_X[i].v;
}

__global__ void cell_switching(int n_cells, Cell* d_X)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    // spot cells become static when u is high
    // if (d_cell_type[i] == 1 && d_X[i].u > 0.5) d_cell_type[i] = 3;
    if (d_pm.tmode == 5) {  // switching for non-advecting/advecting spot
                            // cells
        float top_y = d_tis_max.y - (0.4 * (d_tis_max.y - d_tis_min.y));
        float bot_y = d_tis_min.y + (0.4 * (d_tis_max.y - d_tis_min.y));
        // printf("top_y: %f, bot_y: %f\n", top_y, bot_y);
        if (d_cell_type[i] == 1 && d_X[i].u > 0.4 && d_X[i].y < top_y &&
            d_X[i].y > bot_y)
            d_cell_type[i] = 3;  // don't switch if still in top 10% of
                                 // tissue
    }
    // if (d_cell_type[i] == 2 && d_X[i].u > 180) {
    //     d_cell_type[i] = 1;  // switch to spot cell if u high
    // }
    // if (d_cell_type[i] == 1 && d_X[i].u < 180) {
    //     d_cell_type[i] = 2;  // switch to non-spot cell if u low
    // }
    // float top_y = d_tis_max.y - (0.2 * (d_tis_max.y - d_tis_min.y));
    // float bot_y = d_tis_min.y + (0.2 * (d_tis_max.y - d_tis_min.y));
    // if (d_X[i].y < top_y && d_X[i].y > bot_y) {
    //     if (d_cell_type[i] == 1 && d_X[i].v > d_pm.vthresh) {
    //         d_cell_type[i] = 3;  // switch to spot cell if u high
    //     }
    //     if (d_cell_type[i] == 3 && d_X[i].v < d_pm.vthresh) {
    //         d_cell_type[i] = 2;  // switch to non-spot cell if u low
    //     }
    // }
}

__global__ void death(
    int n_cells, Cell* d_X, int* d_n_cells, curandState* d_state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    float r = curand_uniform(&d_state[i]);

    // if (d_X[i].u > 0.4 && d_X[i].u < 0.43 &&
    if (d_X[i].u > d_pm.u_death &&
        (d_cell_type[i] == 1 ||
            d_cell_type[i] == 3)) {       // die if type 1/3 and u high
        int n = atomicSub(d_n_cells, 1);  // decrement d_n_cells
        // overwrite cell i with last cell in d_X, stop if only one cell left
        if (i < n) {
            d_X[i] = d_X[n - 1];  // copy properties of last cell to cell i
            d_W[i] = d_W[n - 1];
            d_cell_type[i] = d_cell_type[n - 1];
            d_mech_str[i] = d_mech_str[n - 1];
            d_in_ray[i] = d_in_ray[n - 1];
        }
    }
}

void init_rays(Mesh& tis, float rays[100][2])  // maximum of 100 rays
{
    // host function for initialising rays
    // float rays[n_ray][2];  // start and end of each ray
    float p_min, p_max;
    if (h_pm.ray_dir == 0) {
        p_min = tis.get_minimum().x;
        p_max = tis.get_maximum().x;
    }
    if (h_pm.ray_dir == 1) {
        p_min = tis.get_minimum().y;
        p_max = tis.get_maximum().y;
    }

    if (h_pm.n_rays < 2) {  // if only one ray, set to middle
        float center = (p_max + p_min) / 2;
        float p1 = center - (h_pm.s_ray * (p_max - p_min) / 2);
        float p2 = center + (h_pm.s_ray * (p_max - p_min) / 2);
        rays[0][0] = p1;  // start of ray either x or y line
        rays[0][1] = p2;  // end of ray either x or y line
    } else {
        for (int i = 0; i < h_pm.n_rays; i++) {
            float step = (p_max - p_min) / (h_pm.n_rays - 1);
            float p1 = p_min + i * step;  // start of ray either x or y line
            float p2 =
                p1 + (h_pm.s_ray * (p_max - p_min));  // scale by tissue size
            // x_pairs.push_back({x1, x2});
            rays[i][0] = p1;
            rays[i][1] = p2;
        }
    }
}

__global__ void advection(int n_cells, const Cell* d_X, Cell* d_dX,
    const float (*rays)[2], int time_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    if (d_cell_type[i] == 1) {  // only type 1 cells advect
        float norm_x = (d_X[i].x - d_tis_min.x) /
                       (d_tis_max.x - d_tis_min.x);  // % along x axis
        float ad_time =
            norm_x * d_pm.cont_time * 0.5;  // wait time proportional to norm_x

        float ad = d_pm.ad_s;  // default advection strength
        if (d_pm.ad_func == 1 && time_step < ad_time) ad = 0;

        if (d_pm.ray_switch) {
            for (int k = 0; k < d_pm.n_rays; ++k) {
                d_in_ray[i] = false;
                float pos;
                if (d_pm.ray_dir == 0) pos = d_X[i].x;
                if (d_pm.ray_dir == 1) pos = d_X[i].y;
                if (pos >= rays[k][0] && pos <= rays[k][1]) {
                    d_in_ray[i] = true;
                    ad = d_pm.soft_ad_s;  // soft_ad if in ray
                    if (d_pm.ad_func == 1 && time_step < ad_time) ad = 0;
                    break;  // A-P time delay
                }
            }
        }
        if (d_pm.ad_dir == 0) d_dX[i].x += ad;  // +ve x direction
        if (d_pm.ad_dir == 1) d_dX[i].y -= ad;  // -ve y direction
    }
    // if (d_cell_type[i] == 1) d_dX[i].x += 0.3;
}

__global__ void wall_forces_new(int n_cells, const Cell* d_X, Cell* d_dX,
    Po_cell* d_wall_nodes, int n_wall_nodes, Mesh_d wall_mesh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    float min_displ = 1e6;  // Initialize to a large value
    float3 closest_nm = {0, 0, 0};
    for (int j = 0; j < n_wall_nodes; j++) {
        float3 r;  // vector from cell to wall node
        r.x = d_X[i].x - d_wall_nodes[j].x;
        r.y = d_X[i].y - d_wall_nodes[j].y;
        r.z = 0;
        // retrieve unit normal vector of wall node
        float3 nm = pol_to_float3(d_wall_nodes[j]);
        nm.x *= -1;
        nm.y *= -1;
        // calculate displacement of cell to the wall dot product
        float displ = (r.x * nm.x) + (r.y * nm.y) + (r.z * nm.z);

        if (displ < min_displ) {
            min_displ = displ;
            closest_nm = nm;
        }
    }
    // // Determine if cell outside fin using ray-casting
    // float3 ray_start = {-1.0, 1.0, 0.0};  // point outside fin mesh
    // float3 ray_end = {d_X[i].x, d_X[i].y, 0.0};

    // bool outside = wall_mesh.test_exclusion(d_X[i]);  // true if outside
    bool inside = test_exclusion(wall_mesh, d_X[i]);
    // printf("Cell %d: min_displ = %f, inside = %d\n", i, min_displ, inside);
    // printf("in%d\n", inside);

    // if (min_displ < 0 && !inside) {  // only if outside and penetrating wall
    if (inside) {
        auto F_mag = fmaxf(-min_displ, 0);  // force magnitude
        d_dX[i].x +=
            closest_nm.x * F_mag;  // force is product of displ and norm vec
        d_dX[i].y += closest_nm.y * F_mag;
    }
}

// __global__ void wall_forces_new(int n_cells, const Cell* d_X, Cell* d_dX,
//     Po_cell* d_wall_nodes, int n_wall_nodes, Mesh_d wall_mesh)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n_cells) return;

//     float min_dist = 1e30f;
//     int closest_j = -1;
//     for (int j = 0; j < n_wall_nodes; j++) {
//         float dx = d_X[i].x - d_wall_nodes[j].x;
//         float dy = d_X[i].y - d_wall_nodes[j].y;
//         float dist = sqrtf(dx * dx + dy * dy);  // or include z if 3D
//         if (dist < min_dist) {
//             min_dist = dist;
//             closest_j = j;
//         }
//     }

//     if (closest_j >= 0) {
//         float3 r;
//         r.x = d_X[i].x - d_wall_nodes[closest_j].x;
//         r.y = d_X[i].y - d_wall_nodes[closest_j].y;
//         r.z = 0;
//         float3 nm = pol_to_float3(d_wall_nodes[closest_j]);
//         nm.x *= -1;
//         nm.y *= -1;
//         float displ = (r.x * nm.x) + (r.y * nm.y) + (r.z * nm.z);

//         bool inside = test_exclusion(wall_mesh, d_X[i]);

//         if (displ < 0 &&
//             !inside) {  // Only if cell is outside (penetrating) the wall
//             auto F_mag = -displ;
//             d_dX[i].x += nm.x * F_mag;
//             d_dX[i].y += nm.y * F_mag;
//         }
//     }
// }

void update_wall_nodes_from_vtk(
    const std::string& filename, Solution<Po_cell, Grid_solver>& wall_nodes)
{
    Vtk_input input{filename};
    input.read_positions(wall_nodes);
    input.read_polarity(wall_nodes);
    // Mesh wall_mesh{filename};
    *wall_nodes.h_n = input.n_points;
    wall_nodes.copy_to_device();
}

int extract_number(const std::string& filename)
{
    // Example filename: DA-1-10_12-07_0_lmk.vtk
    // Want to extract the '0' before '_lmk.vtk'
    std::regex re("_(\\d+)_[^_]+\\.vtk$");
    std::smatch match;
    if (std::regex_search(filename, match, re)) { return std::stoi(match[1]); }
    return -1;  // or throw/handle error
}

int tissue_sim(int argc, char const* argv[], int walk_id = 0, int step = 0)
{
    std::cout << std::fixed << std::setprecision(6);  // float precision

    // Prepare Random Variable for the Implementation of the Wiener Process
    curandState* d_state;  // define the random number generator on the GPU
    cudaMalloc(&d_state, h_pm.n_max * sizeof(curandState));  // GPU mem alloc
    auto seed = time(NULL);  // random number seed from machine time
    setup_rand_states<<<(h_pm.n_max + 32 - 1) / 32, 32>>>(
        h_pm.n_max, seed, d_state);  // configuring the random number generator
                                     // on the GPU (provided by utils.cuh)

    /* create host variables*/
    // you first create an instance of the Property class on the host, then
    // you connect it to the global variable defined on the device with
    Property<Cell> W{h_pm.n_max, "wiener_process"};  // weiner process
    cudaMemcpyToSymbol(d_W, &W.d_prop, sizeof(d_W));
    Property<float> mech_str{h_pm.n_max, "mech_str"};
    cudaMemcpyToSymbol(d_mech_str, &mech_str.d_prop, sizeof(d_mech_str));
    Property<int> cell_type{h_pm.n_max, "cell_type"};  // cell type labels
    cudaMemcpyToSymbol(d_cell_type, &cell_type.d_prop, sizeof(d_cell_type));
    Property<bool> in_ray{h_pm.n_max, "in_ray"};  // whether cell in ray or not
    cudaMemcpyToSymbol(d_in_ray, &in_ray.d_prop, sizeof(d_in_ray));
    cudaMemcpyToSymbol(d_pm, &h_pm, sizeof(Pm));  // copy host params

    // Solver
    // Solution<Cell, Gabriel_solver> cells{h_pm.n_max, h_pm.g_size,
    // h_pm.r_max}; args are n_max, grid_size, cube_size, gabriel_coefficient
    Solution<Cell, Grid_solver> cells{h_pm.n_max, h_pm.g_size, h_pm.r_max};
    Solution<Po_cell, Grid_solver> wall_nodes{h_pm.wn_max};  // for wall nodes
    // *cells.h_n = h_pm.n_0;

    float rays[h_pm.n_rays][2];  // initialise rays with default values
    for (int i = 0; i < h_pm.n_rays; i++) {
        rays[i][0] = 0;
        rays[i][1] = 0;
    }
    float (*d_rays)[2];
    cudaMalloc(&d_rays, h_pm.n_rays * 2 * sizeof(float));  // GPU mem alloc

    if (h_pm.tmode == 0) {
        random_disk_z(h_pm.init_dist, cells);
        for (int i = 0; i < h_pm.n_0; i++) {
            cell_type.h_prop[i] = (std::rand() % 100 < h_pm.A_init)
                                      ? 1   // randomly assign a proportion of
                                      : 2;  // initial cells with each type
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
                                      ? 2   // set cell type to 1 for spot
                                      : 1;  // cells, and 2 for all others
        }
    }
    if (h_pm.tmode ==
        3) {  // cut the tissue mesh out of a random cloud of cells
        Mesh tis{"../inits/shape3_mesh_3D.vtk"};
        tis.rescale(h_pm.tis_s);  // expand the mesh to fit to the boundaries
        auto tis_min = tis.get_minimum();
        auto tis_max = tis.get_maximum();
        cudaMemcpyToSymbol(d_tis_min, &tis_min, sizeof(float3));  // tis min
        cudaMemcpyToSymbol(d_tis_max, &tis_max, sizeof(float3));  // tis max
        random_rectangle(
            h_pm.init_dist, tis.get_minimum(), tis.get_maximum(), cells);
        auto new_n =
            thrust::remove_if(thrust::host, cells.h_X, cells.h_X + *cells.h_n,
                [&tis](Cell x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            cell_type.h_prop[i] = (std::rand() % 100 < h_pm.A_init)
                                      ? 1   // set cell type to 1 for spot
                                      : 2;  // cells, and 2 for all others
        }
    }
    if (h_pm.tmode == 4) {  // cut the fin mesh out of a random cloud of cells
        Mesh tis{"../inits/shape2_mesh_3D.vtk"};
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
                [&tis](Cell x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            // spot cells appear in leftmost 10% of tissue
            if (cells.h_X[i].x < tis.get_minimum().x + (x_len * 0.1))
                cell_type.h_prop[i] = (std::rand() % 100 < 50) ? 1 : 2;
            else
                cell_type.h_prop[i] = 2;
        }
        init_rays(tis, rays);
        // Print the values of rays after initialization
        std::cout << "Rays after initialization:" << std::endl;
        for (int i = 0; i < h_pm.n_rays; i++) {
            std::cout << "Ray " << i << ": (" << rays[i][0] << ", "
                      << rays[i][1] << ")" << std::endl;
        }
    }
    if (h_pm.tmode == 5) {  // fin with spot aggregation at top
        Mesh tis{"../inits/shape3_mesh_3D.vtk"};
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
                [&tis](Cell x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            // spot cells appear in topmost 10% of tissue
            if (cells.h_X[i].y > tis.get_maximum().y - (y_len * 0.1))
                cell_type.h_prop[i] = (std::rand() % 100 < 50) ? 1 : 2;
            else
                cell_type.h_prop[i] = 2;
        }
        init_rays(tis, rays);
        // Print the values of rays after initialization
        std::cout << "Rays after initialization:" << std::endl;
        for (int i = 0; i < h_pm.n_rays; i++) {
            std::cout << "Ray " << i << ": (" << rays[i][0] << ", "
                      << rays[i][1] << ")" << std::endl;
        }
    }


    for (int i = 0; i < h_pm.n_0; i++) {  // initialise chemical amounts
        cells.h_X[i].u = (std::rand()) / (RAND_MAX + 1.);
        cells.h_X[i].v = (std::rand()) / (RAND_MAX + 1.);
        // cells.h_X[i].u = 0;
        // cells.h_X[i].v = 0;
        // Mesh tis{"../inits/shape1_mesh_3D.vtk"};
        // tis.rescale(h_pm.tis_s);
        // auto y_len = tis.get_maximum().y - tis.get_minimum().y;
        // if (cells.h_X[i].y > tis.get_maximum().y - (y_len * 0.1)) {
        //     cells.h_X[i].u = (std::rand()) / (RAND_MAX + 1.);
        //     cells.h_X[i].v = (std::rand()) / (RAND_MAX + 1.);
        // }
    }

    // Initialise properties and k with zeroes
    for (int i = 0; i < h_pm.n_max; i++) {  // initialise with zeroes
        mech_str.h_prop[i] = 0;
        in_ray.h_prop[i] = false;
    }

    // Initialise the wall nodes from file
    std::vector<std::string> wall_files;
    for (const auto& entry :  // collect all wall file names
        std::filesystem::directory_iterator("../data/lmk_DA-1-10_12-09-25/")) {
        if (entry.path().extension() != ".vtk") continue;
        wall_files.push_back(entry.path().string());
    }
    std::sort(wall_files.begin(), wall_files.end(),  // sort by stage number
        [](const std::string& a, const std::string& b) {
            return extract_number(a) < extract_number(b);
        });
    for (const auto& file : wall_files)
        std::cout << "Wall file: " << file << std::endl;

    // Vtk_input input{"../inits/shape3_mesh_2D.vtk"};
    Vtk_input input{wall_files[0]};    // read in first wall file
    input.read_positions(wall_nodes);  // read in wall nodes from a file
    input.read_polarity(wall_nodes);   // read in wall node polarity
    *wall_nodes.h_n = input.n_points;
    // Mesh_d wall_mesh{
    //     "../inits/shape3_mesh_3D.vtk"};  // for testing if cells are outside
    // fin
    // wall_mesh.copy_to_device();

    std::vector<Triangle_d> host_facets =
        read_facets_from_vtk("../inits/shape3_mesh_3D.vtk");
    for (const auto& facet : host_facets) {
        std::cout << "Facet vertices: (" << facet.V0.x << ", " << facet.V0.y
                  << ", " << facet.V0.z << "), (" << facet.V1.x << ", "
                  << facet.V1.y << ", " << facet.V1.z << "), (" << facet.V2.x
                  << ", " << facet.V2.y << ", " << facet.V2.z << ")\n";
    }

    printf("Number of facets read: %zu\n", host_facets.size());


    Triangle_d* d_facets;
    cudaMalloc(&d_facets, host_facets.size() * sizeof(Triangle_d));
    cudaMemcpy(d_facets, host_facets.data(),
        host_facets.size() * sizeof(Triangle_d), cudaMemcpyHostToDevice);

    Mesh_d mesh_d;
    mesh_d.d_facets = d_facets;
    mesh_d.n_facets = host_facets.size();

    int time_step;  // declare  outside main loop for access in gen_func
    auto generic_function = [&](const int n, const Cell* __restrict__ d_X,
                                Cell* d_dX) {  // then set the mechanical forces
                                               // to zero on the device
        // Set these properties to zero after every timestep so they
        // don't accumulate Called every timesetep, allows you to add
        // custom forces at every timestep e.g. advection
        thrust::fill(thrust::device, mech_str.d_prop,
            mech_str.d_prop + cells.get_d_n(), 0.0);
        thrust::fill(thrust::device, in_ray.d_prop,
            in_ray.d_prop + cells.get_d_n(), false);

        // return wall_forces<Cell, boundary_force>(n, d_X, d_dX, 0);
        if (h_pm.adv_switch)
            advection<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_X, d_dX, d_rays, time_step);
        if (h_pm.fin_walls)
            // return wall_forces_mult<Cell, boundary_forces_mult>(n, d_X,
            // d_dX,
            // 0,
            //     h_pm.w_off_s);  //, num_walls, wall_normals,
            // wall_offsets);
            wall_forces_new<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_X, d_dX, wall_nodes.d_X, *wall_nodes.h_n,
                mesh_d);
    };

    cells.copy_to_device();
    wall_nodes.copy_to_device();
    mech_str.copy_to_device();
    cell_type.copy_to_device();
    in_ray.copy_to_device();

    Vtk_output output{
        "out_" + std::to_string(walk_id) + "_" + std::to_string(step)};
    // create instance of Vtk_output class
    Vtk_output wall_output{
        "out_wall_" + std::to_string(walk_id) + "_" + std::to_string(step)};


    /* the neighbours are initialised with 0. However, you want to use them
       in the proliferation function, which is called first.
        1. proliferation
        2. noise
        3. take_step
       we use a trick, such that the very first call of the proliferation is
       not launched on zeros. here instead of dt we pass 0.0, so that we
       count cells, but do not compute any replacements in the tissue
       -> x[t+1] = x[t] + 0.0 * (dx);
    */

    cells.take_step<pairwise_force>(0.0, generic_function);

    // write out initial condition
    cells.copy_to_host();
    // wall_nodes.copy_to_host();
    mech_str.copy_to_host();
    cell_type.copy_to_host();
    in_ray.copy_to_host();
    wall_nodes.copy_to_host();

    output.write_positions(cells);
    output.write_property(mech_str);
    output.write_property(cell_type);
    output.write_property(in_ray);
    output.write_field(cells, "u", &Cell::u);  // write u of each cell to vtk
    output.write_field(cells, "v", &Cell::v);
    wall_output.write_positions(wall_nodes);
    wall_output.write_polarity(wall_nodes);


    // Main simulation loop
    for (time_step = 0; time_step <= h_pm.cont_time; time_step++) {
        if (time_step > 0 && time_step % 100 == 0) {
            // std::string vtk_filename = wall_files[int(time_step / 100)];
            std::string vtk_filename = wall_files[0];
            std::cout << "Updating wall nodes from: " << vtk_filename
                      << std::endl;
            update_wall_nodes_from_vtk(vtk_filename, wall_nodes);
        }
        for (float T = 0.0; T < 1.0; T += h_pm.dt) {
            generate_noise<<<(cells.get_d_n() + 32 - 1) / 32, 32>>>(
                cells.get_d_n(),
                d_state);  // generate random noise which we will use later
                           // on to move the cells
            if (h_pm.prolif_switch)
                proliferation<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), d_state, cells.d_X, cells.d_old_v,
                    cells.d_n);  // simulate proliferation
            if (h_pm.type_switch)
                cell_switching<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), cells.d_X);  // switch cell types if
            // conditions are metq
            cells.take_step<pairwise_force, friction_on_background>(
                h_pm.dt, generic_function);
            if (h_pm.death_switch)
                death<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), cells.d_X, cells.d_n, d_state);
        }

        if (time_step % int(h_pm.cont_time / h_pm.no_frames) == 0) {
            cells.copy_to_host();
            mech_str.copy_to_host();
            cell_type.copy_to_host();
            in_ray.copy_to_host();
            wall_nodes.copy_to_host();

            output.write_positions(cells);
            output.write_property(mech_str);
            output.write_property(cell_type);
            output.write_property(in_ray);
            output.write_field(cells, "u", &Cell::u);
            output.write_field(cells, "v", &Cell::v);
            wall_output.write_positions(wall_nodes);
            wall_output.write_polarity(wall_nodes);
        }
    }
    return 0;
}

// compile tissue_sim as main when this file is not included as library
// elsewhere
#ifndef COMPILE_AS_LIBRARY
int main(int argc, char const* argv[])
{
    int walk_id = 0, step = 0;
    if (argc > 1) walk_id = std::atoi(argv[1]);
    if (argc > 2) step = std::atoi(argv[2]);
    return tissue_sim(argc, argv, walk_id, step);
}
#endif