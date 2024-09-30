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

const auto r_max = 1.2f;                        // Max contact distance between cells
const auto n_0 = 500;                           // Initial number of cells
const auto force_factor = 3.0f;                 // Arbitrary factor multiplying all forces
const auto r_0 = 0.7f;

const auto cont_time = 3000.0f;                  // Simulation duration in arbitrary time units 1000 = 40h ; 750 = 30h
const auto dt = 0.1f;                           // Time step for Euler integration

// Macro that builds the cell variable type
// MAKE_PT(Cell, u, v); // float3 i .x .y .z .u .v .whatever

template<typename Pt>
__device__ Pt pairwise_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    // This will be only useful in simulations with a wall and a ghost node
    if (i == j){
        return dF;
    }
    if (dist > r_max) return dF;

    auto F = force_factor * (fmaxf(r_0 - dist, 0) - fmaxf(dist - r_0, 0));

    dF.x += r.x * F / dist;
    dF.y += r.y * F / dist;
    dF.z += r.z * F / dist;

    return dF;
}

int main(int argc, char const* argv[])
{
    Solution<float3, Gabriel_solver> cells{n_0, 50, r_max};
    *cells.h_n = n_0;
    random_sphere(0.7, cells);

    cells.copy_to_device();
        
    Vtk_output output{"relaxation"};

    // Main simulation loop
    for (auto time_step = 0; time_step <= cont_time; time_step++) {
        for (auto T = 0.0; T < 1.0; T+=dt) {
            cells.take_step<pairwise_force, friction_on_background>(dt);    
        }

        if(time_step % 10 == 0){
            cells.copy_to_host();
            output.write_positions(cells);
        }
    }
    return 0;
}
