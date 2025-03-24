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

#include <iterator>

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
__device__ bool* d_in_ray;  // whether a cell is in a ray

template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    if (dist > r_max) return dF;  // set cutoff for computing interaction forces

    if (i == j) {      // if the cell is interacting with itself
        dF += d_W[i];  // add stochasticity from the weiner process to the
                       // attributes of the cells

        // Chemical production and degredation

        if (cmode == 0) {  // chemical production and degredation
            dF.u =
                k_prod * (1.0 - Xi.u) *
                (d_cell_type[i] == 1 ||
                    d_cell_type[i] == 3);  // cell type 1/3 produces chemical u
            dF.v = k_prod * (1.0 - Xi.v) *
                   (d_cell_type[i] == 2);  // cell type 2 produces chemical v
            dF.u -= k_deg * (Xi.u);
            dF.v -= k_deg * (Xi.v);
        }

        if (cmode == 1) {
            // see Schnackenberg 1979 eq. 41
            // dF.u = (Xi.u * Xi.u * Xi.v) - Xi.u + a_u;
            // dF.v = -(Xi.u * Xi.u * Xi.v) + b_v;
            // dF.u = (Xi.u * Xi.u * Xi.v) - Xi.u + 2;
            // dF.v = 0.1 - (Xi.u * Xi.u * Xi.v);
            dF.u = (Xi.u * Xi.u * Xi.v) - Xi.u + (Xi.x * 0.1);
            dF.v = -(Xi.u * Xi.u * Xi.v) + (Xi.y * 0.1);
        }
        return dF;
    }

    // Diffusion
    dF.u = -D_u * r.u;
    dF.v = -D_v * r.v;
    // dF.u = -Xi.x * r.u * 0.01;
    // dF.v = -Xi.y * r.v * 0.01;
    // dF.u = -1 * r.u;
    // dF.v = -40 * r.v;
    // dF.u = -0.01 * r.u;
    // dF.v = -0.05 * r.v;


    // Mechanical forces

    if (!mov_switch) return dF;  // if cell movement is off, return no forces

    // default adhesion and repulsion vals for cell interactions
    float Adh = Add;
    float adh = add;
    float Rep = Rdd;
    float rep = rdd;

    if (diff_adh_rep) {
        if ((d_cell_type[i] == 1 and d_cell_type[j] == 1) or
            (d_cell_type[i] == 3 and d_cell_type[j] == 3) or
            (d_cell_type[i] == 1 and d_cell_type[j] == 3) or
            (d_cell_type[i] == 3 and d_cell_type[j] == 1)) {
            Adh = Aii;  // A-A interact with different adh and rep vals
            adh = aii;
            Rep = Rii;
            rep = rii;
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

    float D = noise;  // the magnitude of random noise - set to 0 for
                      // deterministic simulation

    // return noise for every attribute of the cell in this case x,y,z
    d_W[i].x = curand_normal(&d_state[i]) * powf(dt, 0.5) * D / dt;
    d_W[i].y = curand_normal(&d_state[i]) * powf(dt, 0.5) * D / dt;
    d_W[i].z = 0;
    d_W[i].u = 0;
    d_W[i].v = 0;
}

__global__ void proliferation(int n_cells, curandState* d_state, Cell* d_X,
    float3* d_old_v, int* d_n_cells)
{
    int i = blockIdx.x * blockDim.x +
            threadIdx.x;  // get the index of the current cell
    if (i >= n_cells)
        return;  // return nothing if the index is greater than n_cells
    if (n_cells >= (n_max * 0.9))
        return;  // return nothing if the no. cells starts to approach the max

    if (d_cell_type[i] == 1) {
        if (d_mech_str[i] > mech_thresh) return;
        // if (d_X[i].u > 0.85) return;
        if (curand_uniform(&d_state[i]) > (A_div * dt)) return;
    }

    if (d_cell_type[i] == 2) {
        if (d_mech_str[i] > mech_thresh) return;
        if (curand_uniform(&d_state[i]) > (B_div * dt)) return;
    }

    if (d_cell_type[i] == 3) {
        if (d_mech_str[i] > mech_thresh) return;
        if (curand_uniform(&d_state[i]) > (A_div * dt)) return;
    }


    int n = atomicAdd(d_n_cells, 1);

    // new cell added to parent at random angle and fixed distance - this method
    // only works for 2D
    float theta = curand_uniform(&d_state[i]) * 2 * M_PI;

    d_X[n].x = d_X[i].x + (div_dist * cosf(theta));
    d_X[n].y = d_X[i].y + (div_dist * sinf(theta));
    d_X[n].z = 0;

    d_old_v[n] = d_old_v[i];

    d_mech_str[n] = 0.0;


    // set child cell types
    if (d_cell_type[i] == 2) {  // type 2 cells (non-spot) can divide into type
                                // 1 cells when pmode is set correctly
        if (pmode == 0) d_cell_type[n] = 2;
        if (pmode == 1)
            d_cell_type[n] =
                (curand_uniform(&d_state[i]) < r_A_birth)
                    ? 1
                    : 2;  // sometimes cell type 2 produces cell type 1 random
                          // birth of cell type 1 is inhibited by chemical u
        if (pmode == 2)
            d_cell_type[n] =
                (curand_uniform(&d_state[i]) < r_A_birth and d_X[i].u < uthresh)
                    ? 1
                    : 2;  // sometimes cell type 2 produces cell type 1 random
                          // birth of cell type 1 is inhibited by chemical u
    }
    if (d_cell_type[i] == 1) d_cell_type[n] = 1;
    if (d_cell_type[i] == 3) d_cell_type[n] = 3;

    // set child cell chemical amounts
    // d_X[n].u = (d_cell_type[n] == 1) ? 1 : 0;     // if the child is type 1,
    // it is given u=1,v=0 if not, u=0,v=1 d_X[n].v = (d_cell_type[n] == 1) ? 0
    // : 1;

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
    if (d_cell_type[i] == 1 && d_X[i].u > 0.5) d_cell_type[i] = 3;
}

__global__ void death(int n_cells, Cell* d_X, int* d_n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    if (d_X[i].u > u_death &&
        (d_cell_type[i] == 1 ||
            d_cell_type[i] == 3)) {       // die if type 1/3 and u high
        int n = atomicSub(d_n_cells, 1);  // decrement d_n_cells
        // overwrite cell i with last cell in d_X, stop if only one cell left
        if (i < n) {
            d_X[i] = d_X[n - 1];  // copy properties of last cell to cell i
            d_cell_type[i] = d_cell_type[n - 1];
            d_mech_str[i] = d_mech_str[n - 1];
        }
    }
}

void init_rays(Mesh& tis, float rays[n_rays][2])
{
    // float rays[n_ray][2];  // start and end of each ray
    float min_x = tis.get_minimum().x;
    float max_x = tis.get_maximum().x;
    float step = (max_x - min_x) / (n_rays - 1);

    for (int i = 0; i < n_rays; i++) {
        float x1 = min_x + i * step;
        float x2 = x1 + s_ray;
        // x_pairs.push_back({x1, x2});
        rays[i][0] = x1;
        rays[i][1] = x2;
    }
}

__global__ void advection(
    int n_cells, const Cell* d_X, Cell* d_dX, const float (*rays)[2])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    if (d_cell_type[i] == 1) {  // only type 1 one cells advect
        float ad = ad_s;        // default advection strength
        if (ray_switch) {
            for (int k = 0; k < n_rays; ++k) {
                d_in_ray[i] = false;
                if (d_X[i].x >= rays[k][0] && d_X[i].x <= rays[k][1]) {
                    d_in_ray[i] = true;
                    ad = soft_ad_s;  // soft_ad if in ray
                    break;
                }
            }
            // if (!d_in_ray[i]) ad = soft_ad_s;  // soft_ad if not in ray
        }
        d_dX[i].x += ad;
    }
}

int tissue_sim(int argc, char const* argv[])
{
    std::cout << std::fixed
              << std::setprecision(6);  // set precision for floats

    /*
    Prepare Random Variable for the Implementation of the Wiener Process
    */
    curandState* d_state;  // define the random number generator on the GPu
    cudaMalloc(&d_state,
        n_max * sizeof(curandState));  // allocate GPU memory according to
                                       // no. cells
    auto seed =
        time(NULL);  // random number seed - coupled to the time on your machine
    setup_rand_states<<<(n_max + 32 - 1) / 32, 32>>>(
        n_max, seed, d_state);  // configuring the random number generator
                                // on the GPU (provided by utils.cuh)

    /* create host variables*/
    // you first create an instance of the Property class on the host, then
    // you connect it to the global variable defined on the device with
    Property<Cell> W{n_max, "wiener_process"};  // weiner process random number
    cudaMemcpyToSymbol(d_W, &W.d_prop, sizeof(d_W));
    Property<float> mech_str{n_max, "mech_str"};
    cudaMemcpyToSymbol(d_mech_str, &mech_str.d_prop, sizeof(d_mech_str));
    Property<int> cell_type{n_max, "cell_type"};  // cell type labels
    cudaMemcpyToSymbol(d_cell_type, &cell_type.d_prop, sizeof(d_cell_type));
    Property<bool> in_ray{n_max, "in_ray"};  // whether cell is in ray or not
    cudaMemcpyToSymbol(d_in_ray, &in_ray.d_prop, sizeof(d_in_ray));

    // Initial conditions
    Solution<Cell, Gabriel_solver> cells{n_max, 50, r_max};  // intialise solver
    *cells.h_n = n_0;

    float rays[n_rays][2];  // initialise rays with default values
    for (int i = 0; i < n_rays; i++) {
        rays[i][0] = 0;
        rays[i][1] = 0;
    }
    // Allocate memory for rays on the device
    float(*d_rays)[2];
    cudaMalloc(&d_rays, n_rays * 2 * sizeof(float));

    if (tmode == 0) {
        random_disk_z(init_dist, cells);  // initialise random disk with mean
                                          // distance between cells of init_dist
        for (int i = 0; i < n_0; i++) {
            cell_type.h_prop[i] = (std::rand() % 100 < A_init)
                                      ? 1
                                      : 2;  // randomly assign a proportion of
                                            // initial cells with each type
        }
    }
    if (tmode == 1) {
        regular_rectangle(init_dist, std::round(std::sqrt(n_0) / 10) * 10,
            cells);  // initialise rectangle specifying the no. cells along
                     // the x axis
        for (int i = 0; i < n_0; i++) {
            cell_type.h_prop[i] = (std::rand() % 100 < A_init) ? 1 : 2;
        }
    }
    if (tmode == 2) {  // rectangle with spots on one end
        auto sp_size = (A_init / 100.0) * n_0;  // calculate no. cells in spot
        regular_rectangle_w_spot(
            sp_size, init_dist, std::round(std::sqrt(n_0) / 10) * 10, cells);
        for (int i = 0; i < n_0; i++) {
            cell_type.h_prop[i] =
                (i < n_0 - sp_size) ? 2 : 1;  // set cell type to 1 for spot
                                              // cells, and 2 for all others
        }
    }
    if (tmode == 3) {  // cut the tissue mesh out of a random cloud of cells
        Mesh tis{"../inits/shape1_mesh_3D.vtk"};
        tis.rescale(tis_s);  // expand the mesh to fit to the boundaries
        random_rectangle(
            init_dist, tis.get_minimum(), tis.get_maximum(), cells);
        auto new_n =
            thrust::remove_if(thrust::host, cells.h_X, cells.h_X + *cells.h_n,
                [&tis](Cell x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < n_0; i++) {  // set cell types
            cell_type.h_prop[i] = (std::rand() % 100 < A_init)
                                      ? 1
                                      : 2;  // set cell type to 1 for spot
                                            // cells, and 2 for all others
        }
    }
    if (tmode == 4) {  // cut the fin mesh out of a random cloud of cells
        Mesh tis{"../inits/shape1_mesh_3D.vtk"};
        tis.rescale(tis_s);
        auto x_len = tis.get_maximum().x - tis.get_minimum().x;
        random_rectangle(
            init_dist, tis.get_minimum(), tis.get_maximum(), cells);
        auto new_n =
            thrust::remove_if(thrust::host, cells.h_X, cells.h_X + *cells.h_n,
                [&tis](Cell x) { return tis.test_exclusion(x); });
        *cells.h_n = std::distance(cells.h_X, new_n);
        for (int i = 0; i < n_0; i++) {  // set cell types
            // spot cells appear in leftmost 10% of tissue
            if (cells.h_X[i].x < tis.get_minimum().x + (x_len * 0.1))
                cell_type.h_prop[i] = (std::rand() % 100 < 50) ? 1 : 2;
            else
                cell_type.h_prop[i] = 2;
        }
        init_rays(tis, rays);
        // Print the values of rays after initialization
        std::cout << "Rays after initialization:" << std::endl;
        for (int i = 0; i < n_rays; i++) {
            std::cout << "Ray " << i << ": (" << rays[i][0] << ", "
                      << rays[i][1] << ")" << std::endl;
        }
    }

    for (int i = 0; i < n_0; i++) {  // initialise chemical amounts
        // cells.h_X[i].u = (std::rand()) / (RAND_MAX + 1.);
        // cells.h_X[i].v = (std::rand()) / (RAND_MAX + 1.);
        cells.h_X[i].u = 0;
        cells.h_X[i].v = 0;
    }

    // Initialise properties and k with zeroes
    for (int i = 0; i < n_max; i++) {  // initialise with zeroes
        mech_str.h_prop[i] = 0;
        in_ray.h_prop[i] = false;
    }

    cudaMemcpy(
        d_rays, rays, n_rays * 2 * sizeof(float), cudaMemcpyHostToDevice);

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
        if (adv_switch)
            advection<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_X, d_dX, d_rays);
        if (fin_walls)
            return wall_forces_mult<Cell, boundary_forces_mult>(n, d_X, d_dX, 0,
                w_off_s);  //, num_walls, wall_normals, wall_offsets);
    };

    cells.copy_to_device();
    mech_str.copy_to_device();
    cell_type.copy_to_device();
    in_ray.copy_to_device();

    Vtk_output output{"out"};  // create instance of Vtk_output class


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
    in_ray.copy_to_host();

    output.write_positions(cells);
    output.write_property(mech_str);
    output.write_property(cell_type);
    output.write_property(in_ray);
    output.write_field(cells, "u", &Cell::u);  // write u of each cell to vtk
    output.write_field(cells, "v", &Cell::v);


    // Main simulation loop
    for (int time_step = 0; time_step <= cont_time; time_step++) {
        for (float T = 0.0; T < 1.0; T += dt) {
            generate_noise<<<(cells.get_d_n() + 32 - 1) / 32, 32>>>(
                cells.get_d_n(),
                d_state);  // generate random noise which we will use later
                           // on to move the cells
            if (prolif_switch)
                proliferation<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), d_state, cells.d_X, cells.d_old_v,
                    cells.d_n);  // simulate proliferation
            if (type_switch)
                cell_switching<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), cells.d_X);  // switch cell types if
            // conditions are met
            cells.take_step<pairwise_force, friction_on_background>(
                dt, generic_function);
            if (death_switch)
                death<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                    cells.get_d_n(), cells.d_X, cells.d_n);
        }

        if (time_step % int(cont_time / no_frames) == 0) {
            cells.copy_to_host();
            mech_str.copy_to_host();
            cell_type.copy_to_host();
            in_ray.copy_to_host();

            output.write_positions(cells);
            output.write_property(mech_str);
            output.write_property(cell_type);
            output.write_property(in_ray);
            output.write_field(cells, "u",
                &Cell::u);  // write the u part of each cell to vtk
            output.write_field(cells, "v", &Cell::v);
        }
    }
    return 0;
}

// only compile main when this file is not included as library elsewhere
#ifndef COMPILE_AS_LIBRARY
int main(int argc, char const* argv[]) { return tissue_sim(argc, argv); }
#endif