// Toy model for accessing cell fate decisions during cancer development

// Compilation
//
// $ nvcc -std=c++14 -arch=sm_86 {"compiler flags"} Limb_model_simulation.cu
// The values for "-std" and "-arch" flags will depend on your version of CUDA and the specific GPU model you have respectively.
// e.g. -std=c++14 works for CUDA version 11.6 and -arch=sm_86 corresponds to the generation of NVIDIA Geforce 30XX cards.
#include "../include/solvers.cuh"
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"  
#include "../include/property.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"


// global simulation parameters
const float r_max = 0.2;                        // Max distance betwen two cells for which they will interact
const int n_max = 200000;                       // Max number of cells
const float c_div = 0.00002;                    // Probability of cell division per iteration
const float noise = 0;//0.5;                        // Magnitude of noise returned by generate_noise
const int cont_time = 10000;                    // Simulation duration in arbitrary time units 1000 = 40h ; 750 = 30h
const float dt = 0.1;                           // Time step for Euler integration

// tissue initialisation
const float init_dist = 0.1;                    // mean distance between cells when initialised
const int n_0 = 500;                            // Initial number of cells

// cell migration parameters
const float rii = 0.02;                         // Length scales for migration forces for iri-iri (in mm)
const float rxx = 0.011;                        // " xan-on-xan (in mm)
const float rxi = 0.02;                         // " xan-on-iri (in mm)
const float rix = 0.02;                         // " iri-on-xan repulsion (in mm)
const float aix = 0.012;                        // " iri-on-xan attraction (in mm)
const float Rii = 0.00124;                      // Repulsion from iri to iri (mm^2/day)
const float Rxx = 0.00055;                      // Repulsion from xan to xan (mm^2/day)
const float Rxi = 0.00274;                      // Repulsion force on iri due to xan (mm^2/day)
const float Rix = 0.00226;                      // Repulsion force on xan due to iri (mm^2/day)
const float Aix = 0.001956;                     // Attraction force on xan due to iri (mm^2/day)

// iridophore birth parameters
const float alpha = 1;                          // short-range signals for iridophore birth
const float beta = 3.5;                         // long-range signals for iridophore birth
const float eta = 6;                            // cap on max number of iridophores that can be in omegaLoc before it becomes too overcrowed for cell birth
// const float eta = eta + 100000*overSwitch;      // effectively turns off max-density constraint if overSwitch = 1
const float iriRand = 0.03;                     // chance of random melanophore birth when no cells in omegaRand
// const float iriRand = iriRand*randSwitch;       // if randSwitch = 0, turns off random birth
// const float iriRand = iriRand*(popSwitch ~= 2); // if no melanophores included, turns off random iridophore birth

// xanthophore birth parameters
const float phi = 1.3;                          // short-range signals for xanthophsore birth
const float psi = 0.7;                          // long-range signals for xanthophore birth
const float kappa = 10;                         // cap on max number of xanthophores that can be in omegaLoc before overcrowding
// const float kappa = kappa + 100000*overSwitch;  // effectively turns off max-density constraint if overSwitch = 1
const float xanRand = 0.005;                    // chance of random xanthophore birth when no cells in omegaRand
// const float xanRand = xanRand*randSwitch;       // if randSwitch = 0, turns off random birth
// const float xanRand = xanRand*(popSwitch ~= 1); // if no xanthophores included, turn off random xanthophore birth



// Macro that builds the cell variable type
// MAKE_PT(Cell, u, v); // float3 i .x .y .z .u .v .whatever

__device__ float* d_mechanical_strain; // define global variable for mechanical strain on the GPU (device)
__device__ int* d_cell_type; // global variable for cell type on the GPU - iridophore=1, xanthophore=2
__device__ float3* d_W; // global variable for random number from Weiner process for stochasticity
__device__ int* d_ngs_type_A; // no. iri cells in neighbourhood
__device__ int* d_ngs_type_B; // no. xan cells in neighbourhood

template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    // This will be only useful in simulations with a wall and a ghost node
    if (i == j){
        dF += d_W[i]; // add stochasticity from the weiner process to the attributes of the cells
        return dF;
    }
    if (dist > r_max) return dF;

    // we define the strength of adhesion and repulsion
    float Adh = 1;
    float adh = 1;
    float Rep = 1;
    float rep = 1;

    if (d_cell_type[i] == 1 and d_cell_type[j] == 1) { // iri -> iri
        Adh = 0;
        adh = 1;
        Rep = Rii;
        rep = rii;
    }
    if (d_cell_type[i] == 1 and d_cell_type[j] == 2) { // iri -> xan
        Adh = Aix;
        adh = aix;
        Rep = Rix;
        rep = rix;
    }
    if (d_cell_type[i] == 2 and d_cell_type[j] == 1) { // xan -> iri
        Adh = 0;
        adh = 1;
        Rep = Rxi;
        rep = rxi;
    }
    if (d_cell_type[i] == 2 and d_cell_type[j] == 2) { // xan -> xan
        Adh = 0;
        adh = 1;
        Rep = Rxx;
        rep = rxx;
    }


    //float F = (k_rep * fmaxf(rep_ulim - dist, 0) - k_adh * fmaxf(dist - adh_llim, 0)); // forces are also dependent on adhesion and repulsion between cell types
    // Volkening et al. 2015 force potential, function in terms of distance in n dimensions
    float term1 = Adh/adh * expf(-dist / adh);
    float term2 = Rep/rep * expf(-dist / rep);
    float F = term1 - term2;
    //printf("%f\n", F);
    d_mechanical_strain[i] += F; // mechanical strain is the sum of forces on the cell

    dF.x -= r.x * F / dist;
    dF.y -= r.y * F / dist;
    //dF.z -= r.z * F / dist;
    dF.z -= 0;

    // count no. each cell type in neighbourhood
    if (d_cell_type[j] == 1) d_ngs_type_A[i] += 1;
    else d_ngs_type_B[i] += 1;


    return dF;
}

__global__ void generate_noise(int n, curandState* d_state) { // Weiner process for Heun's method
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float D = noise; // the magnitude of random noise - set to 0 for deterministic simulation

    // return noise for every attribute of the cell in this case x,y,z
    d_W[i].x = curand_normal(&d_state[i]) * powf(dt, 0.5) * D / dt;
    d_W[i].y = curand_normal(&d_state[i]) * powf(dt, 0.5) * D / dt;
    //d_W[i].z = curand_normal(&d_state[i]) * powf(dt, 0.5) * D / dt;
    d_W[i].z = 0;
}

__global__ void proliferation(int n_cells, curandState* d_state, float3* d_X, float3* d_old_v, int* d_n_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // get the index of the current cell
    if (i >= n_cells) return; // return nothing if the index is greater than n_cells
    if (n_cells >= (n_max * 0.9)) return;  // return nothing if the no. cells starts to approach the max

    // conditions on iri for division
    if (d_cell_type[i] == 1 and d_ngs_type_A[i] < alpha * d_ngs_type_B[i]) return; // short range condition
    if (d_cell_type[i] == 1 and d_ngs_type_A[i] + d_ngs_type_B[i] > eta) return; // overcrowding condition
    
    // conditions on xan for division
    if (d_cell_type[i] == 2 and d_ngs_type_B[i] < phi * d_ngs_type_A[i]) return; // short range condition
    if (d_cell_type[i] == 2 and d_ngs_type_A[i] + d_ngs_type_B[i] > kappa) return; // overcrowding condition

    float rnd = curand_uniform(&d_state[i]);
    
    if (rnd > (c_div * dt)) return;

    int n = atomicAdd(d_n_cells, 1);

    float theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    float phi = curand_uniform(&d_state[i]) * 2 * M_PI;

    d_X[n].x = d_X[i].x + 0.8 / 4 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + 0.8 / 4 * sinf(theta) * sinf(phi);
    d_X[n].z = 0;

    d_old_v[n] = d_old_v[i];

    d_mechanical_strain[n] = 0.0;
    d_cell_type[n] = 1;
}



int main(int argc, char const* argv[])
{

    /*
    Prepare Random Variable for the Implementation of the Wiener Process
    */
    curandState* d_state; // define the random number generator on the GPu
    cudaMalloc(&d_state, n_max*sizeof(curandState)); // allocate GPU memory according to the number of cells
    auto seed = time(NULL); // random number seed - coupled to the time on your machine
    setup_rand_states<<<(n_max + 32 - 1)/32, 32>>>(n_max, seed, d_state); // configuring the random number generator on the GPU (provided by utils.cuh)

    /* create host variables*/
    // Wiener process
    Property<float3> W{n_max, "wiener_process"}; // define a property for the weiner process
    cudaMemcpyToSymbol(d_W, &W.d_prop, sizeof(d_W)); // connect the global property defined on the GPU to the property defined in this function

    // Mechanical strain
    Property<float> mechanical_strain{n_max, "mech_str"}; // create an instance of the property
    cudaMemcpyToSymbol(d_mechanical_strain, &mechanical_strain.d_prop, sizeof(d_mechanical_strain)); // connect the above instance (on the host) to the global variable on the device

    // No. iri in neighbourhood
    Property<int> ngs_type_A{n_max, "ngs_type_A"}; // create an instance of the property
    cudaMemcpyToSymbol(d_ngs_type_A, &ngs_type_A.d_prop, sizeof(d_ngs_type_A));

    // No. xan in neighbourhood
    Property<int> ngs_type_B{n_max, "ngs_type_B"}; // create an instance of the property
    cudaMemcpyToSymbol(d_ngs_type_B, &ngs_type_B.d_prop, sizeof(d_ngs_type_B));

    // Cell type labels
    Property<int> cell_type{n_max, "cell_type"};
    cudaMemcpyToSymbol(d_cell_type, &cell_type.d_prop, sizeof(d_cell_type));
    for (int i =0; i < n_0; i++) {
        cell_type.h_prop[i] = std::rand() % 2 + 1; // assign each cell randomly the label 1 or 2
    }
    /**/

    // Initial conditions
    
    Solution<float3, Gabriel_solver> cells{n_max, 50, r_max};
    *cells.h_n = n_0;
    //random_sphere(0.7, cells);
    //random_disk_z(init_dist, cells);
    volk_zebra_2D(init_dist, cells);
    for (int i = 0; i < n_0; i++) {
        //printf("ypos: %.6f\n", cells.h_X[i].y);
        if (cells.h_X[i].y == 0.5) {
            cell_type.h_prop[i] = 1;
        } else {
            cell_type.h_prop[i] = 2;
        }
    }



    // Initialise properties with zeroes
    for (int i = 0; i < n_max; i++) { //initialise with zeroes, for loop step size is set to 1 with i++
        mechanical_strain.h_prop[i] = 0;
        ngs_type_A.h_prop[i] = 0;
        ngs_type_B.h_prop[i] = 0;
    }

    auto generic_function = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) { // then set the mechanical forces to zero on the device
        // Set these properties to zero after every timestep so they don't accumulate
        thrust::fill(thrust::device, mechanical_strain.d_prop, mechanical_strain.d_prop + cells.get_d_n(), 0.0);
        thrust::fill(thrust::device, ngs_type_A.d_prop, ngs_type_A.d_prop + cells.get_d_n(), 0);
	    thrust::fill(thrust::device, ngs_type_B.d_prop, ngs_type_B.d_prop + cells.get_d_n(), 0);
    };

    cells.copy_to_device();
    mechanical_strain.copy_to_device();
    cell_type.copy_to_device();
    ngs_type_A.copy_to_device();
    ngs_type_B.copy_to_device();

        
    Vtk_output output{"out"};

    /* the neighbours are initialised with 0. However, you want to use them in the proliferation function, which is called first.
	1. proliferation
	2. noise
	3. take_step
       we use a trick, such that the very first call of the proliferation is not launched on zeros.
       here instead of dt we pass 0.0, so that we count cells, but do not compute any replacements in the tissue
       -> x[t+1] = x[t] + 0.0 * (dx);
    */

    cells.take_step<pairwise_force>(0.0, generic_function);

    // Main simulation loop
    for (int time_step = 0; time_step <= cont_time; time_step++) {
        for (float T = 0.0; T < 1.0; T+=dt) {
            proliferation<<<(cells.get_d_n() + 128 - 1)/128, 128>>>(cells.get_d_n(), d_state, cells.d_X, cells.d_old_v, cells.d_n); // simulate proliferation
            generate_noise<<<(cells.get_d_n() + 32 - 1)/32, 32>>>(cells.get_d_n(), d_state); // generate random noise which we will use later on to move the cells
            cells.take_step<pairwise_force, friction_on_background>(dt, generic_function);    
        }

        if(time_step % 100 == 0){
            cells.copy_to_host();
            mechanical_strain.copy_to_host();
            cell_type.copy_to_host();
            output.write_positions(cells);
            output.write_property(mechanical_strain);
            output.write_property(cell_type);
        }
    }
    return 0;
}