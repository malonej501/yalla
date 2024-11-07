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

// N.B. distances are in millimeters so 0.001 = 1 micrometer

// global simulation parameters
const float r_max = 0.1;                        // Max distance betwen two cells for which they will interact - set to upper bound of donut
const int n_max = 200000;                       // Max number of cells
const float noise = 0;//0.5;                        // Magnitude of noise returned by generate_noise
const int cont_time = 1000;                    // Simulation duration in arbitrary time units 1 = 1 day
const float dt = 0.1;                           // Time step for Euler integration
const int no_frames = 100;                      // no. frames of simulation output to vtk - divide the simulation time by this number

// tissue initialisation
const float init_dist = 0.05;//0.082;                    // mean distance between cells when initialised 
const float div_dist = 0.01;
const int n_0 = 1000;//450;//500;//350;                            // Initial number of cells n.b. this number needs to divide properly between stripes if using volk initial condition
const float A_init = 0;                         // % of the initial cell population that will be type 1 / A

// cell migration parameters
const bool diff_adh_rep = true;                // set to false to turn off differential adhesion and repulsion
// const float rii = 0.02;                         // Length scales for migration forces for iri-iri (in mm)
// const float Rii = 0.00124;                      // Repulsion from iri to iri (mm^2/day)
// const float aii = 0.012;
// const float Aii = 0.001956;

// const float rii = 0.01;                         // Length scales for migration forces for iri-iri (in mm)
// const float Rii = 0.002;                      // Repulsion from iri to iri (mm^2/day)
// const float aii = 0.011;
// const float Aii = 0.0019;

const float rii = 0.012;                         // Length scales for migration forces for iri-iri (in mm)
const float Rii = 0.0045;                      // Repulsion from iri to iri (mm^2/day)
const float aii = 0.019;
const float Aii = 0.0019;

// proliferation parameters
const float A_div = 0.012;                      // 0.02 works well if you have the overcrowding condition
const float B_div = 0.012;                   
const float r_A_birth = 0.08;                   //chance of iridophore birth from background cell
const int Acrowd = 5;                           // max no. A cells in the local disc of A cells before proliferation stops
const int Bcrowd = 5;                           // max no. B cells in the local disc of B cells before proliferation stops
const float uthresh = 0.015;                      // B cells will not change to A if the amount of u exceeds this value

// chemical diffusion rates - this is Fick's first law?
const float D_u = 1.0;
const float D_v = 0.01;

// Macro that builds the cell variable type - instead of type float3 we are making a instance of Cell with attributes x,y,z,u,v where u and v are diffusible chemicals
//MAKE_PT(Cell); // float3 i .x .y .z .u .v .whatever
// to use MAKE_PT(Cell) replace every instance of float3 with Cell
MAKE_PT(Cell, u, v);

__device__ float* d_mechanical_strain; // define global variable for mechanical strain on the GPU (device)
__device__ int* d_cell_type; // global variable for cell type on the GPU - iridophore=1, xanthophore=2, DEAD=0
__device__ Cell* d_W; // global variable for random number from Weiner process for stochasticity
__device__ int* d_ngs_type_A; // no. iri cells in neighbourhood
__device__ int* d_ngs_type_B; // no. xan cells in neighbourhood

template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    //if (dist > r_max) return dF; // Gabriel solver doesn't account for distance when computing neighbourhood, we need to exclude distant pairs
    if (dist > r_max) return dF; // set cutoff for computing forces


    // This will be only useful in simulations with a wall and a ghost node
    if (i == j){
        dF += d_W[i]; // add stochasticity from the weiner process to the attributes of the cells

        // each cell type has a base line production rate of chemical u or v depending on cell type
        float k_prod = 0.3;
        // dF.u += k_prod * (d_cell_type[i] == 1); // cell type 1 produces chemical u
        // dF.v += k_prod * (d_cell_type[i] == 2); // cell type 2 produces chemical v
        dF.u = k_prod * (1.0 - Xi.u) * (d_cell_type[i] ==1);
        dF.v = k_prod * (1.0 - Xi.v) * (d_cell_type[i] ==2);


        // add degredation not dependent on anything
        float k_deg = 0.03;
        dF.u -= k_deg * (Xi.u);
        dF.v -= k_deg * (Xi.v);

        return dF;
    }
    
    dF.u = -D_u * r.u; // r.u is the difference in chemical concentration between cells in pair
    dF.v = -D_v * r.v;


    if (dist < 0.075) { // the radius of the inner disc
        // count no. each cell type in neighbourhood
        if (d_cell_type[j] == 1) d_ngs_type_A[i] += 1;
        else d_ngs_type_B[i] += 1;
    }



    // we define the default strength of adhesion and repulsion
    float Adh = 0;
    float adh = 1;
    float Rep = Rii;
    float rep = rii;

    if (diff_adh_rep) {
        if (d_cell_type[i] == 1 and d_cell_type[j] == 1) { // iri -> iri
            Adh = Aii;
            adh = aii;
            Rep = Rii;
            rep = rii;
        }
    }


    // float F = (k_rep * fmaxf(0.08 - dist, 0) - k_adh * fmaxf(dist - 0.08, 0)); // forces are also dependent on adhesion and repulsion between cell types
    // float F = (Adh * r.x * exp(-sqrt(r.x^2 + r.y^2) / adh)) / (adh * sqrt(r.x^2 + r.y^2)) - (Rep * r.x * exp(-sqrt(r.x^2 - r.y^2) / rep) / (rep * sqrt(r.x^2 - r.y^2)));
    // Volkening et al. 2015 force potential, function in terms of distance in n dimensions
    float term1 = Adh/adh * expf(-dist / adh);
    float term2 = Rep/rep * expf(-dist / rep);
    float F = term1 - term2;
    //printf("%f\n", F);
    d_mechanical_strain[i] -= F; // mechanical strain is the sum of forces on the cell

    dF.x -= r.x * F / dist;
    dF.y -= r.y * F / dist;
    dF.z -= 0;  


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
    d_W[i].u = 0;
    d_W[i].v = 0;
}

__global__ void proliferation(int n_cells, curandState* d_state, Cell* d_X, float3* d_old_v, int* d_n_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // get the index of the current cell
    if (i >= n_cells) return; // return nothing if the index is greater than n_cells
    if (n_cells >= (n_max * 0.9)) return;  // return nothing if the no. cells starts to approach the max

    if (d_cell_type[i] == 1) {
        // if (d_ngs_type_A[i] + d_ngs_type_B[i] > eta) return;
        if (d_ngs_type_A[i] > Acrowd) return;
        if (curand_uniform(&d_state[i]) > (A_div * dt)) return;
    }

    if (d_cell_type[i] == 2) {
        //if (d_ngs_type_A[i] + d_ngs_type_B[i] < eta) return;
        // if (d_ngs_type_A[i] + d_ngs_type_B[i] > eta) return;
        //if (d_ngs_type_A[i] + d_ngs_type_B[i] > ABcrowd) return;
        if (d_ngs_type_B[i] > Bcrowd) return;
        if (curand_uniform(&d_state[i]) > (B_div * dt)) return;
    }
    

    int n = atomicAdd(d_n_cells, 1);

    // new cell added next to parent at random angle
    float theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    float phi = curand_uniform(&d_state[i]) * 2 * M_PI;

    d_X[n].x = d_X[i].x + div_dist / 2 * sinf(theta) * cosf(phi);
    d_X[n].y = d_X[i].y + div_dist / 2 * sinf(theta) * sinf(phi);
    d_X[n].z = 0;

    d_old_v[n] = d_old_v[i];

    d_mechanical_strain[n] = 0.0;
     
    
    // set child cell types    
    if (d_cell_type[i] == 2) {
        d_cell_type[n] = (curand_uniform(&d_state[i]) < r_A_birth and d_X[i].u < uthresh) ? 1 : 2; // sometimes cell type 2 produces cell type 1 random birth of cell type 1 is inhibited by chemical u
    }
    if (d_cell_type[i] == 1) {
        d_cell_type[n] = 1;
    }

    // set child cell chemical amounts
    // d_X[n].u = (d_cell_type[n] == 1) ? 1 : 0;     // if the child is type 1, it is given u=1,v=0 if not, u=0,v=1
    // d_X[n].v = (d_cell_type[n] == 1) ? 0 : 1;

     // half the amount of each chemical upon cell division in the parent cell
    d_X[i].u *= 0.5;
    d_X[i].v *= 0.5;
    // the child inherits the other half of the amount of the chemical
    d_X[n].u = d_X[i].u;
    d_X[n].v = d_X[i].v;
 
}


int main(int argc, char const* argv[])
{

    std::cout << std::fixed << std::setprecision(6); // set precision for floats

    // Print the parameters
    std::cout << "Global Simulation Parameters:\n";
    std::cout << "r_max = " << r_max << "\n";
    std::cout << "n_max = " << n_max << "\n";
    std::cout << "noise = " << noise << "\n";
    std::cout << "cont_time = " << cont_time << "\n";
    std::cout << "dt = " << dt << "\n";
    std::cout << "no_frames = " << no_frames << "\n\n";

    std::cout << "Tissue Initialization:\n";
    std::cout << "init_dist = " << init_dist << "\n";
    std::cout << "div_dist = " << div_dist << "\n";
    std::cout << "n_0 = " << n_0 << "\n";
    std::cout << "A_init = " << A_init << "\n\n";

    std::cout << "Cell Migration Parameters:\n";
    std::cout << "diff_adh_rep = " << (diff_adh_rep ? "true" : "false") << "\n";
    std::cout << "rii = " << rii << "\n";
    std::cout << "Rii = " << Rii << "\n";
    std::cout << "aii = " << aii << "\n";
    std::cout << "Aii = " << Aii << "\n\n";

    std::cout << "Proliferation Parameters:\n";
    std::cout << "A_div = " << A_div << "\n";
    std::cout << "B_div = " << B_div << "\n";
    std::cout << "r_A_birth = " << r_A_birth << "\n";
    std::cout << "Acrowd = " << Acrowd << "\n";
    std::cout << "Bcrowd = " << Bcrowd << "\n";
    std::cout << "uthresh = " << uthresh << "\n\n";

    std::cout << "Chemical Diffusion Rates:\n";
    std::cout << "D_u = " << D_u << "\n";
    std::cout << "D_v = " << D_v << "\n\n";


    /*
    Prepare Random Variable for the Implementation of the Wiener Process
    */
    curandState* d_state; // define the random number generator on the GPu
    cudaMalloc(&d_state, n_max*sizeof(curandState)); // allocate GPU memory according to the number of cells
    auto seed = time(NULL); // random number seed - coupled to the time on your machine
    setup_rand_states<<<(n_max + 32 - 1)/32, 32>>>(n_max, seed, d_state); // configuring the random number generator on the GPU (provided by utils.cuh)

    /* create host variables*/
    // Wiener process
    Property<Cell> W{n_max, "wiener_process"}; // define a property for the weiner process
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
        cell_type.h_prop[i] = (std::rand() % 100 < A_init) ?  1 : 2; //randomly assign a proportion of initial cells with each type
    }

    // for (int i =0; i < n_0; i++) {
    //     cell_type.h_prop[i] = 2; // set all initial cells to be background
    // }

    
    /**/

    // Initial conditions
    
    Solution<Cell, Gabriel_solver> cells{n_max, 50, r_max};
    // Solution<Cell, Grid_solver> cells{n_max, 50, r_max}; //originally using r_max*5
    *cells.h_n = n_0;
    //random_sphere(0.7, cells);
    random_disk_z(init_dist, cells);
    // regular_rectangle(init_dist, std::round(std::sqrt(n_0) / 10) * 10, cells); //initialise square with nx=n_0/2 center will be at (y,x) = (1,1)

    // initialise chemical amounts 
    for (int i = 0; i < n_0; i++) {
        // cells.h_X[i].u = 0; //h_X is host cell
        // cells.h_X[i].v = 0;
        if (cell_type.h_prop[i] == 1) {
            cells.h_X[i].u = 1;
            cells.h_X[i].v = 0;
        } else if (cell_type.h_prop[i] == 2) {
            cells.h_X[i].u = 0;
            cells.h_X[i].v = 1;
        }
    }
    
    // Initialise properties and k with zeroes
    for (int i = 0; i < n_max; i++) { //initialise with zeroes, for loop step size is set to 1 with i++
        mechanical_strain.h_prop[i] = 0;
        ngs_type_A.h_prop[i] = 0;
        ngs_type_B.h_prop[i] = 0;
    }

    auto generic_function = [&](const int n, const Cell* __restrict__ d_X, Cell* d_dX) { // then set the mechanical forces to zero on the device
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
    
    // write out initial condition
    cells.copy_to_host();
    mechanical_strain.copy_to_host();
    cell_type.copy_to_host();
    ngs_type_A.copy_to_host();
    ngs_type_B.copy_to_host();

    output.write_positions(cells);
    output.write_property(mechanical_strain);
    output.write_property(cell_type);
    output.write_property(ngs_type_A);
    output.write_property(ngs_type_B);
    output.write_field(cells, "u", &Cell::u); //write the u part of each cell to vtk
    output.write_field(cells, "v", &Cell::v);


    // Main simulation loop
    for (int time_step = 0; time_step <= cont_time; time_step ++) {
        for (float T = 0.0; T < 1.0; T+=dt) {
            generate_noise<<<(cells.get_d_n() + 32 - 1)/32, 32>>>(cells.get_d_n(), d_state); // generate random noise which we will use later on to move the cells
            proliferation<<<(cells.get_d_n() + 128 - 1)/128, 128>>>(cells.get_d_n(), d_state, cells.d_X, cells.d_old_v, cells.d_n); // simulate proliferation
            cells.take_step<pairwise_force, friction_on_background>(dt, generic_function);
        }

        if(time_step % int(cont_time / no_frames) == 0){
            cells.copy_to_host();
            mechanical_strain.copy_to_host();
            cell_type.copy_to_host();
            ngs_type_A.copy_to_host();
            ngs_type_B.copy_to_host();

            output.write_positions(cells);
            output.write_property(mechanical_strain);
            output.write_property(cell_type);
            output.write_property(ngs_type_A);
            output.write_property(ngs_type_B);
            output.write_field(cells, "u", &Cell::u); //write the u part of each cell to vtk
            output.write_field(cells, "v", &Cell::v);
        }
    }
    return 0;
}