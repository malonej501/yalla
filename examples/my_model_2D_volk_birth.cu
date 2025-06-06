// Toy model for accessing cell fate decisions during cancer development

// Compilation
//
// $ nvcc -std=c++14 -arch=sm_86 {"compiler flags"} Limb_model_simulation.cu
// The values for "-std" and "-arch" flags will depend on your version of CUDA
// and the specific GPU model you have respectively. e.g. -std=c++14 works for
// CUDA version 11.6 and -arch=sm_86 corresponds to the generation of NVIDIA
// Geforce 30XX cards.
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"

// N.B. distances are in millimeters so 0.001 = 1 micrometer

// global simulation parameters
const float r_max = 0.2;    // Max distance betwen two cells for which they will
                            // interact - set to upper bound of donut
const int n_max = 200000;   // Max number of cells
const float c_div = 0.008;  // Probability of cell division per iteration
// const float c_die = 0.05;                   // Probability of cell death per
// iteration
const float noise = 0;  // 0.5;                        // Magnitude of noise
                        // returned by generate_noise
const int cont_time =
    1000;            // Simulation duration in arbitrary time units 1 = 1 day
const float dt = 1;  // Time step for Euler integration
const int no_frames = 100;  // no. frames of simulation output to vtk - divide
                            // the simulation time by this number
const int n_new_cells =
    300;  // no. cells to add to staging after each simulation iteration

// tissue initialisation
const float init_dist =
    0.082;  // 0.082;                    // mean distance between cells when
            // initialised - set to distance between xanthophore and melanophore
const float A_dist = 0.05;   // mean distance between iri-iri
const float B_dist = 0.036;  // mean distance between xan-xan
const int n_0 =
    700;  // Initial number of cells, 50 of each type and 300 of each staging

// cell migration parameters
const bool diff_adh_rep =
    true;  // set to false to turn off differential adhesion and repulsion
const float rii =
    0.02;  // Length scales for migration forces for iri-iri (in mm)
const float rxx = 0.011;     // " xan-on-xan (in mm)
const float rxi = 0.02;      // " xan-on-iri (in mm)
const float rix = 0.02;      // " iri-on-xan repulsion (in mm)
const float aix = 0.012;     // " iri-on-xan attraction (in mm)
const float Rii = 0.00124;   // Repulsion from iri to iri (mm^2/day)
const float Rxx = 0.00055;   // Repulsion from xan to xan (mm^2/day)
const float Rxi = 0.00274;   // Repulsion force on iri due to xan (mm^2/day)
const float Rix = 0.00226;   // Repulsion force on xan due to iri (mm^2/day)
const float Aix = 0.001956;  // Attraction force on xan due to iri (mm^2/day)

// iridophore birth parameters
const float alpha = 1;   // short-range signals for iridophore birth
const float beta = 3.5;  // long-range signals for iridophore birth
const float eta =
    6;  // cap on max number of iridophores that can be in omegaLoc before it
        // becomes too overcrowed for cell birth
// const float eta = eta + 100000*overSwitch;      // effectively turns off
// max-density constraint if overSwitch = 1
const float iriRand =
    0.03;  // chance of random melanophore birth when no cells in omegaRand
// const float iriRand = iriRand*randSwitch;       // if randSwitch = 0, turns
// off random birth const float iriRand = iriRand*(popSwitch ~= 2); // if no
// melanophores included, turns off random iridophore birth

// xanthophore birth parameters
const float phi = 1.3;   // short-range signals for xanthophore birth
const float psi = 1.2;   // long-range signals for xanthophore birth
const float kappa = 10;  // cap on max number of xanthophores that can be in
                         // omegaLoc before overcrowding
// const float kappa = kappa + 100000*overSwitch;  // effectively turns off
// max-density constraint if overSwitch = 1
const float xanRand =
    0.005;  // chance of random xanthophore birth when no cells in omegaRand
// const float xanRand = xanRand*randSwitch;       // if randSwitch = 0, turns
// off random birth const float xanRand = xanRand*(popSwitch ~= 1); // if no
// xanthophores included, turn off random xanthophore birth

// cell death parameters
const float xi = 1.2;  // long-range signals for xanthophore death
const float iriProb =
    0.03;  // chance of iridophores death due to long-range interactions
// const float iriProb = iriProb*deathSwitch*(longSwitch == 0);        // if
// deathSwitch = 0 or longSwitch = 1, sets probability of melanophore death due
// to long-range interactions to 0


// Macro that builds the cell variable type - instead of type float3 we are
// making a instance of Cell with attributes x,y,z,u,v where u and v are
// diffusible chemicals
MAKE_PT(Cell, u, v);  // float3 i .x .y .z .u .v .whatever

__device__ float* d_mechanical_strain;  // define global variable for mechanical
                                        // strain on the GPU (device)
__device__ int* d_cell_type;  // global variable for cell type on the GPU -
                              // iridophore=1, xanthophore=2, DEAD=0, staging=-1
__device__ Cell* d_W;  // global variable for random number from Weiner process
                       // for stochasticity
__device__ int* d_ngs_type_A;   // no. iri cells in neighbourhood
__device__ int* d_ngs_type_B;   // no. xan cells in neighbourhood
__device__ int* d_ngs_type_Ac;  // no. iri cells in overcrowded neighbourhood
__device__ int* d_ngs_type_Bc;  // no. xan cells in overcrowded neighbourhood
__device__ int* d_ngs_type_Ad;  // no. iri cells in donut neighbourhood
__device__ int* d_ngs_type_Bd;  // no. xan cells in donut neighbourhood

template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    // This will be only useful in simulations with a wall and a ghost node
    if (i == j) {
        dF += d_W[i];  // add stochasticity from the weiner process to the
                       // attributes of the cells
        return dF;
    }

    // counting cells in different regions
    if ((dist > 0.318) and (dist < 0.318 + 0.025)) {  // count cells in donut
        if (d_cell_type[j] == 1)
            d_ngs_type_Ad[i] += 1;
        else
            d_ngs_type_Bd[i] += 1;
    }
    // printf("%f\n", d_ngs_type_Ad[i], d_ngs_type_Bd[i]);
    if (dist < 0.082) {  // count cells in overcrowding region
        if (d_cell_type[j] == 1)
            d_ngs_type_Ac[i] += 1;
        else
            d_ngs_type_Bc[i] += 1;
    }
    if (dist < 0.075) {  // the radius of the inner disc for cell proliferation
                         // conditions
        // count no. each cell type in neighbourhood
        if (d_cell_type[j] == 1)
            d_ngs_type_A[i] += 1;
        else
            d_ngs_type_B[i] += 1;
    }

    // if (dist > r_max) return dF; // Gabriel solver doesn't account for
    // distance when computing neighbourhood, we need to exclude distant pairs
    if (dist > r_max) return dF;  // set cutoff for computing forces


    // we define the default strength of adhesion and repulsion
    float Adh = 1;
    float adh = 1;
    float Rep = 1;
    float rep = 1;

    if (diff_adh_rep) {
        if (d_cell_type[i] == 1 and d_cell_type[j] == 1) {  // iri -> iri
            Adh = 0;
            adh = 1;
            Rep = Rii;
            rep = rii;
        }
        if (d_cell_type[i] == 1 and d_cell_type[j] == 2) {  // iri -> xan
            Adh = Aix;
            adh = aix;
            Rep = Rix;
            rep = rix;
        }
        if (d_cell_type[i] == 2 and d_cell_type[j] == 1) {  // xan -> iri
            Adh = 0;
            adh = 1;
            Rep = Rxi;
            rep = rxi;
        }
        if (d_cell_type[i] == 2 and d_cell_type[j] == 2) {  // xan -> xan
            Adh = 0;
            adh = 1;
            Rep = Rxx;
            rep = rxx;
        }
    }
    if (d_cell_type[i] == 0 and
        d_cell_type[j] ==
            0) {  // dead -> dead, so dead cells achieve a relaxed state
        Adh = 0;
        adh = 1;
        Rep = Rix;
        rep = rix;
    }
    // } else {
    //     Adh = Aix;
    //     adh = aix;
    //     Rep = Rix;
    //     rep = rix;
    // }

    // Volkening et al. 2015 force potential, function in terms of distance in n
    // dimensions
    float term1 = Adh / adh * expf(-dist / adh);
    float term2 = Rep / rep * expf(-dist / rep);
    float F = term1 - term2;
    // printf("%f\n", F);
    d_mechanical_strain[i] -=
        F;  // mechanical strain is the sum of forces on the cell

    dF.x -= r.x * F / dist;
    dF.y -= r.y * F / dist;
    dF.z -= 0;

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
    // d_W[i].z = curand_normal(&d_state[i]) * powf(dt, 0.5) * D / dt;
    d_W[i].z = 0;
}

__device__ void overwrite_cell(int i, int n, Cell* d_X, int* d_cell_type,
    float* d_mechanical_strain, int* d_ngs_type_A, int* d_ngs_type_B,
    int* d_ngs_type_Ac, int* d_ngs_type_Bc, int* d_ngs_type_Ad,
    int* d_ngs_type_Bd)
{
    if (i < n) {
        d_X[i] = d_X[n - 1];  // Copy properties of the last cell to cell i
        d_cell_type[i] = d_cell_type[n - 1];
        d_mechanical_strain[i] = d_mechanical_strain[n - 1];
        d_ngs_type_A[i] = d_ngs_type_A[n - 1];
        d_ngs_type_B[i] = d_ngs_type_B[n - 1];
        d_ngs_type_Ac[i] = d_ngs_type_Ac[n - 1];
        d_ngs_type_Bc[i] = d_ngs_type_Bc[n - 1];
        d_ngs_type_Ad[i] = d_ngs_type_Ad[n - 1];
        d_ngs_type_Bd[i] = d_ngs_type_Bd[n - 1];
    }
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

    if (!(d_cell_type[i] == -1 || d_cell_type[i] == -2))
        return;  // skip unless a cell is of staging type

    // if all conditions are met, the cell is moved from the staging area into
    // the actual tissue, if not it is moved to the dead area
    if (d_cell_type[i] == -1) {
        if (d_ngs_type_A[i] <
                alpha * d_ngs_type_B[i] and  // short range condition 1
            d_ngs_type_B[i] <
                beta * d_ngs_type_A[i] and  // short range condition 2
            d_ngs_type_Ac[i] + d_ngs_type_Bc[i] > eta  // overcrowding condition
        ) {
            d_cell_type[i] = 1;
            return;
        }
    }
    if (d_cell_type[i] == -2) {
        if (d_ngs_type_B[i] < phi * d_ngs_type_A[i] and
            d_ngs_type_A[i] < psi * d_ngs_type_B[i] and
            d_ngs_type_Ac[i] + d_ngs_type_Bc[i] > kappa) {
            d_cell_type[i] = 2;
            return;
        }
    }
    // printf("Cell should die");

    // d_X[i].z -= r_max * 10;  // when cells die, pop them out by 10 times the
    //                          // maximum interaction distance
    // d_cell_type[i] = 0;      // set cell type to 0 which means dead
    int n = atomicSub(d_n_cells, 1);
    overwrite_cell(i, n, d_X, d_cell_type, d_mechanical_strain, d_ngs_type_A,
        d_ngs_type_B, d_ngs_type_Ac, d_ngs_type_Bc, d_ngs_type_Ad,
        d_ngs_type_Bd);
}

// __global__ void incrementCells(int* d_n_cells) {
//     atomicAdd(d_n_cells, 1);
// }

// void stage_new_cells(int N, cells) {
//     for (int i = 0; i < N; i++) {
//         auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 2);
//         auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;

//         incrementCells<<<1, 1>>>(cells.d_n_cells);
//         cells.d_X[n].x = r * sin(phi);
//         cells.d_X[n].y = r * cos(phi);
//         cells.d_X[n].z = 0;

//         cells.d_old_v[n] = cells.d_old_v[i];

//         cells.d_cell_type[n] = -1;
//     }
// }

__global__ void stage_new_cells(int n_cells, curandState* d_state, Cell* d_X,
    float3* d_old_v, int* d_n_cells, int n_new_cells)
{
    int i = blockIdx.x * blockDim.x +
            threadIdx.x;  // get the index of the current cell
    if (i >= n_cells)
        return;  // return nothing if the index is greater than n_cells
    if (n_cells >= (n_max * 0.9))
        return;  // return nothing if the no. cells starts to approach the max

    // auto R_MAX = pow(n_cells / 0.9069, 1. / 2) * init_dist / 2;
    auto R_MAX = pow(100 / 0.9069, 1. / 2) * init_dist / 2;
    if (i < n_new_cells * 2) {
        int n = atomicAdd(d_n_cells, 1);
        auto r = R_MAX * pow(curand_uniform(&d_state[i]), 1. / 2);
        float PHI = curand_uniform(&d_state[i]) * 2 * M_PI;
        d_X[n].x = r * sin(PHI);
        d_X[n].y = r * cos(PHI);
        d_X[n].z = 0;

        d_old_v[n] = d_old_v[i];
        // d_cell_type[n] = -1;
        if (i < n_new_cells)
            d_cell_type[n] = -1;  // assign cell type -1 or -2 randomly for
                                  // staging type 1 or type 2 respectively
        else
            d_cell_type[n] = -2;
        // d_cell_type[n] = int(curand_uniform(&d_state[i])) - 2;
    }
}

__global__ void death(int n_cells, curandState* d_state, Cell* d_X,
    float3* d_old_v, int* d_n_cells)
{
    int i = blockIdx.x * blockDim.x +
            threadIdx.x;  // get the index of the current cell
    if (i >= n_cells)
        return;  // return nothing if the index is greater than n_cells
    if (n_cells >= (n_max * 0.9))
        return;  // return nothing if the no. cells starts to approach the max

    if (d_cell_type[i] == 0) return;  // cells that are already dead cannot die

    float rnd = curand_uniform(&d_state[i]);

    // if (rnd > (c_die * dt)) return; // die with probability c_die * dt

    // iridophore death condition
    if (d_cell_type[i] == 1 and d_ngs_type_B[i] < d_ngs_type_A[i])
        return;  // if no. xan in nbhd doesn't exceed no. iri, don't die

    // xanthophore death condition
    if (d_cell_type[i] == 2 and d_ngs_type_A[i] < d_ngs_type_B[i])
        return;  // if no. iri in nbhd doesn't exceed no. xan, don't die

    // long range iridophore death condition
    if (d_cell_type[i] == 1 and d_ngs_type_Ad[i] < xi * d_ngs_type_Bd[i] and
        (rnd > iriProb))
        return;  // if the no. irid is less than the no. xan in the donut, don't
                 // die
    // added rnd > iriProb from matlab code

    // d_X[i].z -= r_max * 10; // when cells die, pop them out by 10 times the
    // maximum interaction distance
    // //printf("Cell index: %d, Cell type: %d, Position: %f\n", i,
    // d_cell_type[i], d_X[i].z); d_cell_type[i] = 0; // set cell type to 0
    // which means dead
    int n = atomicSub(d_n_cells, 1);
    overwrite_cell(i, n, d_X, d_cell_type, d_mechanical_strain, d_ngs_type_A,
        d_ngs_type_B, d_ngs_type_Ac, d_ngs_type_Bc, d_ngs_type_Ad,
        d_ngs_type_Bd);
}


int main(int argc, char const* argv[])
{
    /*
    Prepare Random Variable for the Implementation of the Wiener Process
    */
    curandState* d_state;  // define the random number generator on the GPu
    cudaMalloc(&d_state,
        n_max * sizeof(curandState));  // allocate GPU memory according to
                                       // the number of cells
    auto seed =
        time(NULL);  // random number seed - coupled to the time on your machine
    setup_rand_states<<<(n_max + 32 - 1) / 32, 32>>>(
        n_max, seed, d_state);  // configuring the random number generator
                                // on the GPU (provided by utils.cuh)

    /* create host variables*/
    // Wiener process
    Property<Cell> W{
        n_max, "wiener_process"};  // define a property for the weiner process
    cudaMemcpyToSymbol(d_W, &W.d_prop,
        sizeof(d_W));  // connect the global property defined on the GPU to
                       // the property defined in this function

    // Mechanical strain
    Property<float> mechanical_strain{
        n_max, "mech_str"};  // create an instance of the property
    cudaMemcpyToSymbol(d_mechanical_strain, &mechanical_strain.d_prop,
        sizeof(d_mechanical_strain));  // connect the above instance (on the
                                       // host) to the global variable on
                                       // the device

    // No. iri in neighbourhood
    Property<int> ngs_type_A{
        n_max, "ngs_type_A"};  // create an instance of the property
    cudaMemcpyToSymbol(d_ngs_type_A, &ngs_type_A.d_prop, sizeof(d_ngs_type_A));
    // No. xan in neighbourhood
    Property<int> ngs_type_B{
        n_max, "ngs_type_B"};  // create an instance of the property
    cudaMemcpyToSymbol(d_ngs_type_B, &ngs_type_B.d_prop, sizeof(d_ngs_type_B));

    // No. iri in overcrowded neighbourhood
    Property<int> ngs_type_Ac{
        n_max, "ngs_type_Ac"};  // create an instance of the property
    cudaMemcpyToSymbol(
        d_ngs_type_Ac, &ngs_type_Ac.d_prop, sizeof(d_ngs_type_Ac));
    // No. xan in overcrowded neighbourhood
    Property<int> ngs_type_Bc{
        n_max, "ngs_type_Bc"};  // create an instance of the property
    cudaMemcpyToSymbol(
        d_ngs_type_Bc, &ngs_type_Bc.d_prop, sizeof(d_ngs_type_Bc));

    // No. iri in donut
    Property<int> ngs_type_Ad{
        n_max, "ngs_type_Ad"};  // create an instance of the property
    cudaMemcpyToSymbol(
        d_ngs_type_Ad, &ngs_type_Ad.d_prop, sizeof(d_ngs_type_Ad));
    // No. xan in donut
    Property<int> ngs_type_Bd{
        n_max, "ngs_type_Bd"};  // create an instance of the property
    cudaMemcpyToSymbol(
        d_ngs_type_Bd, &ngs_type_Bd.d_prop, sizeof(d_ngs_type_Bd));

    // Cell type labels
    Property<int> cell_type{n_max, "cell_type"};
    cudaMemcpyToSymbol(d_cell_type, &cell_type.d_prop, sizeof(d_cell_type));
    for (int i = 0; i < n_0; i++) {
        // cell_type.h_prop[i] = std::rand() % 2 + 1; // assign each cell
        // randomly the label 1 or 2
        if (i < 50) cell_type.h_prop[i] = 1;
        if (i >= 50 and i < 100) cell_type.h_prop[i] = 2;
        if (i >= 100 and i < 400) cell_type.h_prop[i] = -1;
        if (i >= 400 and i < 700) cell_type.h_prop[i] = -2;
    }
    /**/

    // Initial conditions

    // Solution<Cell, Gabriel_solver> cells{n_max, 50, r_max};
    //  Solution<Cell, Grid_solver> cells{n_max, 100, r_max*5}; //originally
    //  using r_max*5
    Solution<Cell, Grid_solver> cells{
        n_max, 50, r_max * 5};  // originally using r_max*5
    *cells.h_n = n_0;
    // random_sphere(0.7, cells);
    random_disk_z(init_dist, cells);
    // regular_rectangle(init_dist, std::round(std::sqrt(n_0) / 10) * 10,
    // cells); //initialise square with nx=n_0/2 center will be at (y,x) =
    // (1,1) volk_zebra_2D(A_dist, cells); for (int i = 0; i < n_0; i++) {
    //     //printf("ypos: %.6f\n", cells.h_X[i].y);
    //     if (cells.h_X[i].y == 0.5) { // this is how you access the
    //     coordinates of each cell - positions are stored in h_X
    //         cell_type.h_prop[i] = 2; // make cells in middle stripe
    //         xanthophores
    //     } else {
    //         cell_type.h_prop[i] = 1; // make everything else iridophores
    //     }
    // }
    // 2 stripe initial condition
    // for (int i = 0; i < n_0; i++) {
    //     //printf("ypos: %.6f\n", cells.h_X[i].y);
    //     //if (cells.h_X[i].x > 0.5 and cells.h_X[i].x <1.5) { // 1
    //     vertical stripe if ((cells.h_X[i].x > 0.5 and cells.h_X[i].x
    //     < 1.5) or (cells.h_X[i].x > 2.5 and cells.h_X[i].x < 3.5)){ //2
    //     vertical stripes
    //         cell_type.h_prop[i] = 1; // make cells iridophores
    //     } else {
    //         cell_type.h_prop[i] = 2; // make everything else xanthophores
    //     }
    // }


    // Initialise properties with zeroes
    for (int i = 0; i < n_max; i++) {  // initialise with zeroes, for loop
                                       // step size is set to 1 with i++
        mechanical_strain.h_prop[i] = 0;
        ngs_type_A.h_prop[i] = 0;
        ngs_type_B.h_prop[i] = 0;
        ngs_type_Ad.h_prop[i] = 0;
        ngs_type_Bd.h_prop[i] = 0;
        ngs_type_Ac.h_prop[i] = 0;
        ngs_type_Bc.h_prop[i] = 0;
    }

    auto generic_function = [&](const int n, const Cell* __restrict__ d_X,
                                Cell* d_dX) {  // then set the mechanical forces
                                               // to zero on the device
        // Set these properties to zero after every timestep so they
        // don't accumulate
        thrust::fill(thrust::device, mechanical_strain.d_prop,
            mechanical_strain.d_prop + cells.get_d_n(), 0.0);
        thrust::fill(thrust::device, ngs_type_A.d_prop,
            ngs_type_A.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, ngs_type_B.d_prop,
            ngs_type_B.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, ngs_type_Ad.d_prop,
            ngs_type_Ad.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, ngs_type_Bd.d_prop,
            ngs_type_Bd.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, ngs_type_Ac.d_prop,
            ngs_type_Ac.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, ngs_type_Bc.d_prop,
            ngs_type_Bc.d_prop + cells.get_d_n(), 0);
    };

    cells.copy_to_device();
    mechanical_strain.copy_to_device();
    cell_type.copy_to_device();
    ngs_type_A.copy_to_device();
    ngs_type_B.copy_to_device();
    ngs_type_Ac.copy_to_device();
    ngs_type_Bc.copy_to_device();
    ngs_type_Ad.copy_to_device();
    ngs_type_Bd.copy_to_device();


    Vtk_output output{"out"};

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

    // export the initial conditions
    cells.copy_to_host();
    mechanical_strain.copy_to_host();
    cell_type.copy_to_host();
    ngs_type_A.copy_to_host();
    ngs_type_B.copy_to_host();
    ngs_type_Ac.copy_to_host();
    ngs_type_Bc.copy_to_host();
    ngs_type_Ad.copy_to_host();
    ngs_type_Bd.copy_to_host();

    output.write_positions(cells);
    output.write_property(mechanical_strain);
    output.write_property(cell_type);
    output.write_property(ngs_type_A);
    output.write_property(ngs_type_B);
    output.write_property(ngs_type_Ad);
    output.write_property(ngs_type_Bd);

    // Main simulation loop
    for (int time_step = 0; time_step <= cont_time; time_step++) {
        for (float T = 0.0; T < 1.0; T += dt) {
            // generate_noise<<<(cells.get_d_n() + 32 - 1)/32,
            // 32>>>(cells.get_d_n(), d_state); // generate random noise
            // which we will use later on to move the cells
            stage_new_cells<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_state, cells.d_X, cells.d_old_v, cells.d_n,
                n_new_cells);
            cells.take_step<pairwise_force>(0.0, generic_function);
            proliferation<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), d_state, cells.d_X, cells.d_old_v,
                cells.d_n);  // simulate proliferation
            cells.take_step<pairwise_force, friction_on_background>(
                dt, generic_function);
            // death<<<(cells.get_d_n() + 128 - 1) / 128,
            // 128>>>(cells.get_d_n(),
            //     d_state, cells.d_X, cells.d_old_v,
            //     cells.d_n);  // simulate death
        }

        if (time_step % 1 == 0) {
            cells.copy_to_host();
            mechanical_strain.copy_to_host();
            cell_type.copy_to_host();
            ngs_type_A.copy_to_host();
            ngs_type_B.copy_to_host();
            ngs_type_Ac.copy_to_host();
            ngs_type_Bc.copy_to_host();
            ngs_type_Ad.copy_to_host();
            ngs_type_Bd.copy_to_host();

            output.write_positions(cells);
            output.write_property(mechanical_strain);
            output.write_property(cell_type);
            output.write_property(ngs_type_A);
            output.write_property(ngs_type_B);
            output.write_property(ngs_type_Ad);
            output.write_property(ngs_type_Bd);
        }
    }
    return 0;
}