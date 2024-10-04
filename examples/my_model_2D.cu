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

const float r_max = 1.2;                        // Max contact distance between cells
const int n_0 = 500;                           // Initial number of cells
const int n_max = 200000;                       // Max number of cells
const float c_div = 0.0005;                      // Probability of cell division per iteration
const float noise = 0.2;                        // Magnitude of noise returned by generate_noise
const float self_adh = 3.0;                     // Strength of adhesion
const float non_self_rep = 4.0;                 // Strength of repulsion
const float rep_ulim = 0.7;                      // The maximum distance between two cells for which they will repel
const float adh_llim = 0.8;                     // The minimum distance between two cells for which they will atract



const int cont_time = 1000;                  // Simulation duration in arbitrary time units 1000 = 40h ; 750 = 30h
const float dt = 0.1;                           // Time step for Euler integration

// Macro that builds the cell variable type
// MAKE_PT(Cell, u, v); // float3 i .x .y .z .u .v .whatever

__device__ float* d_mechanical_strain; // define global variable for mechanical strain on the GPU (device)
__device__ int* d_cell_type; // global variable for cell type on the GPU
__device__ float3* d_W; // global variable for random number from Weiner process for stochasticity

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

    float k_adh = (d_cell_type[i] == d_cell_type[j]) ? self_adh : 1.0; // if the cell types are the same set adhesion to 3.0 if not then 1.0
    float k_rep = (d_cell_type[i] == d_cell_type[j]) ? 1.0 : non_self_rep;


    float F = (k_adh * fmaxf(rep_ulim - dist, 0) - k_rep * fmaxf(dist - adh_llim, 0)); // forces are also dependent on adhesion and repulsion between cell types
    // printf("%f\n", F);
    d_mechanical_strain[i] += F; // mechanical strain is the sum of forces on the cell

    dF.x += r.x * F / dist;
    dF.y += r.y * F / dist;
    //dF.z += r.z * F / dist;
    dF.z += 0;

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
    if (n_cells >= (n_max * 0.9)) return;

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

    Property<float3> W{n_max, "wiener_process"}; // define a property for the weiner process
    cudaMemcpyToSymbol(d_W, &W.d_prop, sizeof(d_W)); // connect the global property defined on the GPU to the property defined in this function

    // Initial conditions
    
    Solution<float3, Gabriel_solver> cells{n_max, 50, r_max};
    *cells.h_n = n_0;
    //random_sphere(0.7, cells);
    random_disk_z(0.7, cells);

    cells.copy_to_device();

    // Mechanical strain

    Property<float> mechanical_strain{n_max, "mech_str"}; // create an instance of the property
    cudaMemcpyToSymbol(d_mechanical_strain, &mechanical_strain.d_prop, sizeof(d_mechanical_strain)); // connect the above instance (on the host) to the global variable on the device

    for (int i = 0; i < n_max; i++) { //initialise with zeroes, for loop step size is set to 1 with i++
        mechanical_strain.h_prop[i] = 0;
    }
    mechanical_strain.copy_to_device();

    auto generic_function = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) { // then set the mechanical forces to zero on the device
        thrust::fill(thrust::device, mechanical_strain.d_prop, mechanical_strain.d_prop + cells.get_d_n(), 0.0);
    };

    // Cell type labels

    Property<int> cell_type{n_max, "cell_type"};
    cudaMemcpyToSymbol(d_cell_type, &cell_type.d_prop, sizeof(d_cell_type));
    for (int i =0; i < n_0; i++) {
        cell_type.h_prop[i] = std::rand() % 2 + 1; // assign each cell randomly the label 1 or 2
    }
    cell_type.copy_to_device();
        
    Vtk_output output{"relaxation"};

    // Main simulation loop
    for (int time_step = 0; time_step <= cont_time; time_step++) {
        for (float T = 0.0; T < 1.0; T+=dt) {
            proliferation<<<(cells.get_d_n() + 128 - 1)/128, 128>>>(cells.get_d_n(), d_state, cells.d_X, cells.d_old_v, cells.d_n); // simulate proliferation
            generate_noise<<<(cells.get_d_n() + 32 - 1)/32, 32>>>(cells.get_d_n(), d_state); // generate random noise which we will use later on to move the cells
            cells.take_step<pairwise_force, friction_on_background>(dt, generic_function);    
        }

        if(time_step % 10 == 0){
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