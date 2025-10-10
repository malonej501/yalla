// Toy model for accessing cell fate decisions during cancer development

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
MAKE_PT(Cell, u, v);

// define global variables for the GPU
__device__ float* d_mech_str;
__device__ int* d_cell_type;  // cell_type: A=1-Iri/Mel, B=2-Xan, DEAD=0
__device__ Cell* d_W;  // random number from Weiner process for stochasticity
__device__ bool* d_in_slow;   // whether a cell is in a slow region
__device__ Pm d_pm;           // simulation parameters (host h_pm)
__device__ float3 d_tis_min;  // min coordinate of tissue mesh
__device__ float3 d_tis_max;  // max coordinate of tissue mesh
__device__ int* d_ngs_A;      // no. spot cells in neighbourhood
__device__ int* d_ngs_B;      // no. non-spot cells in neighbourhood
__device__ int* d_ngs_Ac;     // overcrowded neighbourhood
__device__ int* d_ngs_Bc;     // overcrowded neighbourhood
__device__ int* d_ngs_Ad;     // donut neighbourhood
__device__ int* d_ngs_Bd;     // donut neighbourhood

template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    // counting cells in different nbhds
    if ((dist > 0.318) and (dist < 0.318 + 0.025)) {  // donut
        if (d_cell_type[j] == 1)
            d_ngs_Ad[i] += 1;
        else
            d_ngs_Bd[i] += 1;
    }
    if (dist < 0.075) {  // overcrowding region
        if (d_cell_type[j] == 1)
            d_ngs_Ac[i] += 1;
        else
            d_ngs_Bc[i] += 1;
    }
    if (dist < 0.075) {  // inner disc for cell proliferation conditions
        if (d_cell_type[j] == 1)
            d_ngs_A[i] += 1;
        else
            d_ngs_B[i] += 1;
    }

    if (dist > d_pm.r_max)  // dist = norm3df(r.x, r.y, r.z) solvers line 308
        return dF;          // cutoff for chemical and mechanical interaction

    if (d_cell_type[i] < 0 || d_cell_type[j] < 0) {
        return dF;  // cells in staging area have no interactions
    }

    if (i == j) {      // if the cell is interacting with itself
        dF += d_W[i];  // add stochasticity from the weiner process to the
                       // attributes of the cells

        // Chemical production and degredation
        if (d_pm.chem_switch) {
            if (d_pm.cmode == 0) {  // chemical production and degredation
                dF.u = d_pm.k_prod * (1.0 - Xi.u) *
                       (d_cell_type[i] == 1 ||
                           d_cell_type[i] == 3);  // cell type 1/3 produce u
                dF.v =
                    d_pm.k_prod * (1.0 - Xi.v) *
                    (d_cell_type[i] == 2);  // cell type 2 produces chemical v
                // dF.u = d_pm.k_prod * ((d_cell_type[i] == 1 || d_cell_type[i]
                // == 3) &
                //                          Xi.u < 1);  // stop making u when it
                //                                      //   reaches 1
                // dF.v = d_pm.k_prod *
                //        ((d_cell_type[i] == 2) & Xi.v < 1);  // stop making v
                //        when
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
                dF.u = lambda * ((f_u * Xi.u * Xi.u) / (1 + f_v * Xi.v) -
                                    m_u * Xi.u + s_u);
                dF.v = lambda * (g_u * Xi.u * Xi.u - m_v * Xi.v);
            }
        }
        return dF;
    }

    // Diffusion
    if (d_pm.chem_switch) {
        dF.u = -d_pm.D_u * r.u;  // r = Xi - Xj solvers.cuh line 448
        dF.v = -d_pm.D_v * r.v;
    }
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
    float Adh = 0;  // d_pm.Add;
    float adh = 0;  // d_pm.add;
    float Rep = 0;  // d_pm.Rdd;
    float rep = 0;  // d_pm.rdd;

    if (d_pm.diff_adh_rep) {
        if (d_cell_type[i] == 1 and d_cell_type[j] == 1) {
            Adh = 0;  // A-A interact with different adh and rep vals
            adh = 1;
            Rep = 0.00124;
            rep = 0.02;
        }
        if (d_cell_type[i] == 2 and d_cell_type[j] == 1) {
            Adh = 0;  // A-A interact with different adh and rep vals
            adh = 1;
            Rep = 0.00274;
            rep = 0.02;
        }
        if (d_cell_type[i] == 1 and d_cell_type[j] == 2) {
            Adh = 0.001956;  // B-B interact with different adh and rep vals
            adh = 0.012;
            Rep = 0.00226;
            rep = 0.02;
        }
        if (d_cell_type[i] == 2 and d_cell_type[j] == 2) {
            Adh = 0;  // A-B interact with different adh and rep vals
            adh = 1;
            Rep = 0.00055;
            rep = 0.011;
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


__global__ void stage_new_cells(int n_cells, curandState* d_state, Cell* d_X,
    float3* d_old_v, int* d_n_cells, Mesh_d wall_mesh)
{
    int i = blockIdx.x * blockDim.x +
            threadIdx.x;  // get the index of the current cell
    if (i >= n_cells)
        return;  // return nothing if the index is greater than n_cells
    if (n_cells >= (d_pm.n_max * 0.9))
        return;  // return nothing if the no. cells starts to approach the
                 // max

    if (i < d_pm.n_new_cells) {  // threads with i < n_new_cell create new cells
        int n = atomicAdd(d_n_cells, 1);
        d_X[n].x = d_tis_min.x +
                   (d_tis_max.x - d_tis_min.x) * curand_uniform(&d_state[i]);
        d_X[n].y = d_tis_min.y +
                   (d_tis_max.y - d_tis_min.y) * curand_uniform(&d_state[i]);
        d_X[n].z = 0;

        d_old_v[n] = d_old_v[i];
        // d_cell_type[n] = -1;
        d_cell_type[n] = (i < (d_pm.n_new_cells / 2)) ? -1 : -2;
        // Remove if outside mesh
        if (test_exclusion(wall_mesh, d_X[n])) {
            d_cell_type[n] = -99;  // Mark for removal
        }
    }
}

__global__ void clean_up(int n_cells, Cell* d_X, int* d_n_cells)
{
    // Remove cells that are marked for death by swapping with last cell.
    // N.B. if n-1 is also dead, a dead cell will remain until the next call,
    // thus this function is called repeatedly until no dead cells remain.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    if (d_cell_type[i] < 0) {             // if the cell is marked for removal
        int n = atomicSub(d_n_cells, 1);  // decrement d_n_cells
        if (i < n) {
            d_X[i] = d_X[n - 1];  // copy properties of last cell to cell i
            d_W[i] = d_W[n - 1];
            d_cell_type[i] = d_cell_type[n - 1];
            d_mech_str[i] = d_mech_str[n - 1];
            // d_old_v[i] = d_old_v[n - 1];
            d_in_slow[i] = d_in_slow[n - 1];
            d_ngs_A[i] = d_ngs_A[n - 1];
            d_ngs_B[i] = d_ngs_B[n - 1];
            d_ngs_Ac[i] = d_ngs_Ac[n - 1];
            d_ngs_Bc[i] = d_ngs_Bc[n - 1];
            d_ngs_Ad[i] = d_ngs_Ad[n - 1];
            d_ngs_Bd[i] = d_ngs_Bd[n - 1];
        }
    }
}

__global__ void proliferation(int n_cells, curandState* d_state, Cell* d_X,
    float3* d_old_v, int* d_n_cells, Mesh_d wall_mesh)
{
    // change cells from staging to active types if conditions are met
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // index of current cell

    if (i >= n_cells) return;                   // stop if i >= n_cells
    if (n_cells >= (d_pm.n_max * 0.9)) return;  // no div above n_max
    // if (d_mech_str[i] > d_pm.mech_thresh) return;   // no div above

    // if (d_cell_type[i] == 1 || d_cell_type[i] == 2) return;

    if (d_cell_type[i] == -1) {
        if (d_ngs_A[i] > d_pm.alpha * d_ngs_B[i] &&   // short range
            d_ngs_Bd[i] > d_pm.beta * d_ngs_Ad[i] &&  // long range
            d_ngs_Ac[i] + d_ngs_Bc[i] < d_pm.eta      // overcrowding

        )
            d_cell_type[i] = 1;
    } else if (d_cell_type[i] == -2) {
        if (d_ngs_B[i] > d_pm.phi * d_ngs_A[i] &&    // short range
            d_ngs_Ad[i] > d_pm.psi * d_ngs_Bd[i] &&  // long range
            d_ngs_Ac[i] + d_ngs_Bc[i] < d_pm.kappa   // overcrowding
        )
            d_cell_type[i] = 2;
    }
}

__global__ void death(
    int n_cells, curandState* d_state, Cell* d_X, int* d_n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    float r = curand_uniform(&d_state[i]);
    if (d_cell_type[i] == -1 || d_cell_type[i] == -2) return;  // don't die

    // short-range competition spot cells
    // if spot and non-spot in nbhd exceed spot, die
    if (d_cell_type[i] == 1 and d_ngs_B[i] > d_ngs_A[i]) {
        d_cell_type[i] = -1;  // mark for death
    }
    // short-range competition non-spot cells
    // if non-spot and no. spot in nbhd  exceeds no. non-spot, die
    if (d_cell_type[i] == 2 and d_ngs_A[i] > d_ngs_B[i]) {
        d_cell_type[i] = -2;
    }
    // long range spot-cell death condition
    // if spot and no. spot in donut exceeds no. non-spot, die
    if (d_cell_type[i] == 1 and d_ngs_Ad[i] > d_pm.xi * d_ngs_Bd[i] and
        (r > d_pm.q_death)) {
        d_cell_type[i] = -1;
    }
}

__global__ void cell_switching(int n_cells, Cell* d_X)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    // spot cells become static when u is high
    // if (d_cell_type[i] == 1 && d_X[i].u > 0.5) d_cell_type[i] = 3;
    // if (d_pm.tmode == 5) {  // switching for non-advecting/advecting spot
    // cells
    //     float top_y = d_tis_max.y - (0.4 * (d_tis_max.y - d_tis_min.y));
    //     float bot_y = d_tis_min.y + (0.4 * (d_tis_max.y - d_tis_min.y));
    //     if (d_cell_type[i] == 1 && d_X[i].u > 0.5 && d_X[i].y < top_y &&
    //         d_X[i].y > bot_y)
    //         d_cell_type[i] = 3;  // don't switch if still in top 10% of
    //         tissue
    // }
    // if (d_cell_type[i] == 2 && d_X[i].u > 180) {
    //     d_cell_type[i] = 1;  // switch to spot cell if u high
    // }
    // if (d_cell_type[i] == 1 && d_X[i].u < 180) {
    //     d_cell_type[i] = 2;  // switch to non-spot cell if u low
    // }
    float top_y = d_tis_max.y - (0.2 * (d_tis_max.y - d_tis_min.y));
    float bot_y = d_tis_min.y + (0.2 * (d_tis_max.y - d_tis_min.y));
    if (d_X[i].y < top_y && d_X[i].y > bot_y) {
        if (d_cell_type[i] == 1 && d_X[i].v > d_pm.vthresh) {
            d_cell_type[i] = 3;  // switch to spot cell if u high
        }
        if (d_cell_type[i] == 3 && d_X[i].v < d_pm.vthresh) {
            d_cell_type[i] = 2;  // switch to non-spot cell if u low
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
    std::cout << "slow region y: " << p1 << " to " << p2 << "\n";
}

__global__ void advection(int n_cells, const Cell* d_X, Cell* d_dX,
    const float* d_slow_reg, int time_step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    // printf("slow_reg: %f, %f\n", (*slow_reg)[0], (*slow_reg)[1]);

    if ((d_X[i].y < d_slow_reg[0]) && (d_X[i].y > d_slow_reg[1])) {
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
    int n_cells, const Cell* d_X, Cell* d_dX, Mesh_d wall_mesh)
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
__global__ void grow_cells(int n_cells, Pt* d_X, float amount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    d_X[i].x *= amount;
    d_X[i].y *= amount;
    // z remains unchanged
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

__global__ void mark_live_cells(
    const int* d_cell_type, int* d_flags, int n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cells) { d_flags[i] = (d_cell_type[i] >= 0) ? 1 : 0; }
}

// 2. Scatter live cells to compact arrays
__global__ void scatter_live_cells(const Cell* d_X_in, Cell* d_X_out,
    const Cell* d_W_in, Cell* d_W_out, const int* d_cell_type_in,
    int* d_cell_type_out, const float* d_mech_str_in, float* d_mech_str_out,
    const bool* d_in_slow_in, bool* d_in_slow_out, const int* d_ngs_A_in,
    int* d_ngs_A_out, const int* d_ngs_B_in, int* d_ngs_B_out,
    const int* d_ngs_Ac_in, int* d_ngs_Ac_out, const int* d_ngs_Bc_in,
    int* d_ngs_Bc_out, const int* d_ngs_Ad_in, int* d_ngs_Ad_out,
    const int* d_ngs_Bd_in, int* d_ngs_Bd_out, const int* d_flags,
    const int* d_scan, int n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cells && d_flags[i]) {
        int new_idx = d_scan[i];
        d_X_out[new_idx] = d_X_in[i];
        d_W_out[new_idx] = d_W_in[i];
        d_cell_type_out[new_idx] = d_cell_type_in[i];
        d_mech_str_out[new_idx] = d_mech_str_in[i];
        d_in_slow_out[new_idx] = d_in_slow_in[i];
        d_ngs_A_out[new_idx] = d_ngs_A_in[i];
        d_ngs_B_out[new_idx] = d_ngs_B_in[i];
        d_ngs_Ac_out[new_idx] = d_ngs_Ac_in[i];
        d_ngs_Bc_out[new_idx] = d_ngs_Bc_in[i];
        d_ngs_Ad_out[new_idx] = d_ngs_Ad_in[i];
        d_ngs_Bd_out[new_idx] = d_ngs_Bd_in[i];
    }
}


void compact_cells(int n_cells, Cell* d_X, Cell* d_W, int* d_cell_type,
    float* d_mech_str, bool* d_in_slow, int* d_ngs_A, int* d_ngs_B,
    int* d_ngs_Ac, int* d_ngs_Bc, int* d_ngs_Ad, int* d_ngs_Bd, int* d_n_cells)
{
    // Allocate temp arrays
    thrust::device_vector<int> d_flags(n_cells);
    thrust::device_vector<int> d_scan(n_cells);

    // Mark live cells
    mark_live_cells<<<(n_cells + 127) / 128, 128>>>(
        d_cell_type, thrust::raw_pointer_cast(d_flags.data()), n_cells);

    // Scan to get new indices
    thrust::exclusive_scan(
        thrust::device, d_flags.begin(), d_flags.end(), d_scan.begin());

    // Get new cell count
    int last_flag = 0, new_n = 0;
    cudaMemcpy(&last_flag,
        thrust::raw_pointer_cast(d_flags.data()) + n_cells - 1, sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(&new_n, thrust::raw_pointer_cast(d_scan.data()) + n_cells - 1,
        sizeof(int), cudaMemcpyDeviceToHost);
    new_n += last_flag;

    // Allocate output arrays
    Cell* d_X_out;
    cudaMalloc(&d_X_out, new_n * sizeof(Cell));
    Cell* d_W_out;
    cudaMalloc(&d_W_out, new_n * sizeof(Cell));
    int* d_cell_type_out;
    cudaMalloc(&d_cell_type_out, new_n * sizeof(int));
    float* d_mech_str_out;
    cudaMalloc(&d_mech_str_out, new_n * sizeof(float));
    bool* d_in_slow_out;
    cudaMalloc(&d_in_slow_out, new_n * sizeof(bool));
    int* d_ngs_A_out;
    cudaMalloc(&d_ngs_A_out, new_n * sizeof(int));
    int* d_ngs_B_out;
    cudaMalloc(&d_ngs_B_out, new_n * sizeof(int));
    int* d_ngs_Ac_out;
    cudaMalloc(&d_ngs_Ac_out, new_n * sizeof(int));
    int* d_ngs_Bc_out;
    cudaMalloc(&d_ngs_Bc_out, new_n * sizeof(int));
    int* d_ngs_Ad_out;
    cudaMalloc(&d_ngs_Ad_out, new_n * sizeof(int));
    int* d_ngs_Bd_out;
    cudaMalloc(&d_ngs_Bd_out, new_n * sizeof(int));

    // Scatter live cells
    scatter_live_cells<<<(n_cells + 127) / 128, 128>>>(d_X, d_X_out, d_W,
        d_W_out, d_cell_type, d_cell_type_out, d_mech_str, d_mech_str_out,
        d_in_slow, d_in_slow_out, d_ngs_A, d_ngs_A_out, d_ngs_B, d_ngs_B_out,
        d_ngs_Ac, d_ngs_Ac_out, d_ngs_Bc, d_ngs_Bc_out, d_ngs_Ad, d_ngs_Ad_out,
        d_ngs_Bd, d_ngs_Bd_out, thrust::raw_pointer_cast(d_flags.data()),
        thrust::raw_pointer_cast(d_scan.data()), n_cells);

    // Copy output arrays back
    cudaMemcpy(d_X, d_X_out, new_n * sizeof(Cell), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_W, d_W_out, new_n * sizeof(Cell), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_cell_type, d_cell_type_out, new_n * sizeof(int),
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mech_str, d_mech_str_out, new_n * sizeof(float),
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_in_slow, d_in_slow_out, new_n * sizeof(bool),
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_ngs_A, d_ngs_A_out, new_n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_ngs_B, d_ngs_B_out, new_n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_ngs_Ac, d_ngs_Ac_out, new_n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_ngs_Bc, d_ngs_Bc_out, new_n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_ngs_Ad, d_ngs_Ad_out, new_n * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_ngs_Bd, d_ngs_Bd_out, new_n * sizeof(int), cudaMemcpyDeviceToDevice);

    // Update cell count
    cudaMemcpy(d_n_cells, &new_n, sizeof(int), cudaMemcpyHostToDevice);

    // Free temp arrays
    cudaFree(d_X_out);
    cudaFree(d_W_out);
    cudaFree(d_cell_type_out);
    cudaFree(d_mech_str_out);
    cudaFree(d_in_slow_out);
    cudaFree(d_ngs_A_out);
    cudaFree(d_ngs_B_out);
    cudaFree(d_ngs_Ac_out);
    cudaFree(d_ngs_Bc_out);
    cudaFree(d_ngs_Ad_out);
    cudaFree(d_ngs_Bd_out);
}


int tissue_sim(int argc, char const* argv[], int walk_id = 0, int step = 0)
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

    /* create host variables*/
    // you first create an instance of the Property class on the host, then
    // you connect it to the global variable defined on the device with
    Property<Cell> W{
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
    Property<int> ngs_A{h_pm.n_max, "ngs_A"};
    cudaMemcpyToSymbol(d_ngs_A, &ngs_A.d_prop, sizeof(d_ngs_A));
    Property<int> ngs_B{h_pm.n_max, "ngs_B"};
    cudaMemcpyToSymbol(d_ngs_B, &ngs_B.d_prop, sizeof(d_ngs_B));
    Property<int> ngs_Ac{h_pm.n_max, "ngs_Ac"};
    cudaMemcpyToSymbol(d_ngs_Ac, &ngs_Ac.d_prop, sizeof(d_ngs_Ac));
    Property<int> ngs_Bc{h_pm.n_max, "ngs_Bc"};
    cudaMemcpyToSymbol(d_ngs_Bc, &ngs_Bc.d_prop, sizeof(d_ngs_Bc));
    Property<int> ngs_Ad{h_pm.n_max, "ngs_Ad"};
    cudaMemcpyToSymbol(d_ngs_Ad, &ngs_Ad.d_prop, sizeof(d_ngs_Ad));
    Property<int> ngs_Bd{h_pm.n_max, "ngs_Bd"};
    cudaMemcpyToSymbol(d_ngs_Bd, &ngs_Bd.d_prop, sizeof(d_ngs_Bd));

    // Initial conditions
    // Solution<Cell, Gabriel_solver> cells{h_pm.n_max, h_pm.g_size,
    // h_pm.r_max};
    Solution<Cell, Grid_solver> cells{h_pm.n_max, h_pm.g_size, h_pm.r_max * 5};
    // args are n_max, grid_size, cube_size
    float slow_reg[2] = {0, 0};  // initialise slow_reg with default values
    float* d_slow_reg;
    cudaMalloc(&d_slow_reg, 2 * sizeof(float));  // GPU mem alloc

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
                [&tis](Cell x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            cell_type.h_prop[i] = (std::rand() % 100 < h_pm.A_init)
                                      ? 1   // set cell type to 1 for spot
                                      : 2;  // cells, and 2 for all others
        }
        update_slow_reg(tis, slow_reg);
        cudaMemcpy(
            d_slow_reg, slow_reg, 2 * sizeof(float), cudaMemcpyHostToDevice);
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
                [&tis](Cell x) { return tis.test_exclusion(x); });
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
                [&tis](Cell x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < h_pm.n_0; i++) {  // set cell types
            // spot cells appear in topmost 10% of tissue
            if (cells.h_X[i].y > tis.get_maximum().y - (y_len * 0.1))
                cell_type.h_prop[i] = (std::rand() % 100 < 50) ? 1 : 2;
            else
                cell_type.h_prop[i] = 2;
        }
        update_slow_reg(tis, slow_reg);
        cudaMemcpy(
            d_slow_reg, slow_reg, 2 * sizeof(float), cudaMemcpyHostToDevice);
    }


    for (int i = 0; i < h_pm.n_0; i++) {  // initialise chemical amounts
        // cells.h_X[i].u = (std::rand()) / (RAND_MAX + 1.);
        // cells.h_X[i].v = (std::rand()) / (RAND_MAX + 1.);
        cells.h_X[i].u = 0;
        cells.h_X[i].v = 0;
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
        in_slow.h_prop[i] = false;
        ngs_A.h_prop[i] = 0;
        ngs_B.h_prop[i] = 0;
        ngs_Ac.h_prop[i] = 0;
        ngs_Bc.h_prop[i] = 0;
        ngs_Ad.h_prop[i] = 0;
        ngs_Bd.h_prop[i] = 0;
    }

    Fin fin("../inits/fin_init.vtk",
        "fin_" + std::to_string(walk_id) + "_" + std::to_string(step));


    int time_step;  // declare  outside main loop for access in gen_func
    auto generic_function = [&](const int n, const Cell* __restrict__ d_X,
                                Cell* d_dX) {  // then set the mechanical
        // forces to zero on the device
        // remove cells marked for death
        // auto new_end = thrust::remove_if(thrust::device,
        // cell_type.d_prop,
        //     cell_type.d_prop + n_cells,
        //     [] __device__(int type) { return type == -1 || type == -2;
        //     });
        // n_cells = new_end - d_X;
        // Set these properties to zero after every timestep so they
        // don't accumulate Called every timesetep, allows you to add
        // custom forces at every timestep e.g. advection
        thrust::fill(thrust::device, mech_str.d_prop,
            mech_str.d_prop + cells.get_d_n(), 0.0);
        thrust::fill(thrust::device, in_slow.d_prop,
            in_slow.d_prop + cells.get_d_n(), false);
        thrust::fill(
            thrust::device, ngs_A.d_prop, ngs_A.d_prop + cells.get_d_n(), 0);
        thrust::fill(
            thrust::device, ngs_B.d_prop, ngs_B.d_prop + cells.get_d_n(), 0);
        thrust::fill(
            thrust::device, ngs_Ac.d_prop, ngs_Ac.d_prop + cells.get_d_n(), 0);
        thrust::fill(
            thrust::device, ngs_Bc.d_prop, ngs_Bc.d_prop + cells.get_d_n(), 0);
        thrust::fill(
            thrust::device, ngs_Ad.d_prop, ngs_Ad.d_prop + cells.get_d_n(), 0);
        thrust::fill(
            thrust::device, ngs_Bd.d_prop, ngs_Bd.d_prop + cells.get_d_n(), 0);

        // return wall_forces<Cell, boundary_force>(n, d_X, d_dX, 0);
        if (h_pm.adv_switch)
            advection<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_X, d_dX, d_slow_reg, time_step);
        if (h_pm.fin_walls)
            wall_forces_new<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_X, d_dX, fin.mesh);
    };

    cells.copy_to_device();
    mech_str.copy_to_device();
    cell_type.copy_to_device();
    in_slow.copy_to_device();
    ngs_A.copy_to_device();
    ngs_B.copy_to_device();
    ngs_Ac.copy_to_device();
    ngs_Bc.copy_to_device();
    ngs_Ad.copy_to_device();
    ngs_Bd.copy_to_device();

    Vtk_output output{
        "out_" + std::to_string(walk_id) + "_" + std::to_string(step)};
    // create instance of Vtk_output class


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
    mech_str.copy_to_host();
    cell_type.copy_to_host();
    in_slow.copy_to_host();
    ngs_A.copy_to_host();
    ngs_B.copy_to_host();
    ngs_Ac.copy_to_host();
    ngs_Bc.copy_to_host();
    ngs_Ad.copy_to_host();
    ngs_Bd.copy_to_host();

    output.write_positions(cells);
    output.write_property(mech_str);
    output.write_property(cell_type);
    output.write_property(in_slow);
    // output.write_field(cells, "u", &Cell::u);  // write u of each cell to
    // vtk output.write_field(cells, "v", &Cell::v);
    output.write_property(ngs_A);
    output.write_property(ngs_B);
    output.write_property(ngs_Ac);
    output.write_property(ngs_Bc);
    output.write_property(ngs_Ad);
    output.write_property(ngs_Bd);

    fin.write_vtk();


    // Main simulation loop
    for (time_step = 0; time_step <= h_pm.cont_time; time_step++) {
        if (h_pm.t_grow_switch && time_step > 0 && time_step % 10 == 0) {
            fin.grow(h_pm.t_growth_rate);  // grow the fin by 10% every
            float3 tis_min = fin.get_minimum();
            float3 tis_max = fin.get_maximum();
            cudaMemcpyToSymbol(d_tis_min, &tis_min, sizeof(float3));
            cudaMemcpyToSymbol(d_tis_max, &tis_max, sizeof(float3));
            // 10 timesteps
            grow_cells<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), cells.d_X, h_pm.t_growth_rate);
            update_slow_reg(fin, slow_reg);
            cudaMemcpy(d_slow_reg, &slow_reg, 2 * sizeof(float),
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
                    cells.get_d_n(), cells.d_X);  // switch cell types if
                                                  // conditions are met

            if (h_pm.death_switch)  // death occurs once per day - 20 days total
                if (time_step % int(h_pm.cont_time / 20) == 0) {
                    death<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                        cells.get_d_n(), d_state, cells.d_X, cells.d_n);
                }
            // Remove cells marked for death, repeat until all removed
            // int prev_n, curr_n;
            // do {
            //     prev_n = cells.get_d_n();
            //     clean_up<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            //         cells.get_d_n(), cells.d_X, cells.d_n);
            //     curr_n = cells.get_d_n();
            // } while (curr_n < prev_n);  // Repeat until cell count stabilizes
            compact_cells(cells.get_d_n(), cells.d_X, W.d_prop,
                cell_type.d_prop, mech_str.d_prop, in_slow.d_prop, ngs_A.d_prop,
                ngs_B.d_prop, ngs_Ac.d_prop, ngs_Bc.d_prop, ngs_Ad.d_prop,
                ngs_Bd.d_prop, cells.d_n);

            cells.take_step<pairwise_force, friction_on_background>(
                h_pm.dt, generic_function);
        }

        if (time_step % int(h_pm.cont_time / h_pm.no_frames) == 0) {
            cells.copy_to_host();
            mech_str.copy_to_host();
            cell_type.copy_to_host();
            in_slow.copy_to_host();
            ngs_A.copy_to_host();
            ngs_B.copy_to_host();
            ngs_Ac.copy_to_host();
            ngs_Bc.copy_to_host();
            ngs_Ad.copy_to_host();
            ngs_Bd.copy_to_host();


            output.write_positions(cells);
            output.write_property(mech_str);
            output.write_property(cell_type);
            output.write_property(in_slow);
            // output.write_field(cells, "u", &Cell::u);
            // output.write_field(cells, "v", &Cell::v);
            output.write_property(ngs_A);
            output.write_property(ngs_B);
            output.write_property(ngs_Ac);
            output.write_property(ngs_Bc);
            output.write_property(ngs_Ad);
            output.write_property(ngs_Bd);
            fin.write_vtk();
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