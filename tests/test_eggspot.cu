#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <iterator>

#include "../examples/eggspot.cu"
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/mesh.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"
#include "../params/params.h"
#include "minunit.cuh"

const char* test_proliferation() {}