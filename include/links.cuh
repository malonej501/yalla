// Links between points, to simulate protrusions, ... Similar models have been
// used in https://dx.doi.org/doi:10.1073/pnas.97.19.10448 and
// https://dx.doi.org/doi:10.1371/journal.pcbi.1004952
#pragma once

#include <assert.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <time.h>

#include <functional>

#include "mesh.cuh"
#include "utils.cuh"


struct Link {
    int a, b;
};

using Check_link = std::function<bool(int a, int b)>;

bool every_link(int a, int b) { return true; }

class Links {
public:
    Link* h_link;
    Link* d_link;
    int* h_n = (int*)malloc(sizeof(int));
    int* d_n;
    const int n_max;
    curandState* d_state;
    float strength;
    Links(int n_max, float strength = 1.f / 5)
        : n_max{n_max}, strength{strength}
    {
        h_link = (Link*)malloc(n_max * sizeof(Link));
        cudaMalloc(&d_link, n_max * sizeof(Link));
        cudaMalloc(&d_n, sizeof(int));
        cudaMalloc(&d_state, n_max * sizeof(curandState));
        *h_n = n_max;
        set_d_n(n_max);
        reset();
        auto seed = time(NULL);
        setup_rand_states<<<(n_max + 32 - 1) / 32, 32>>>(n_max, seed, d_state);
    }
    ~Links()
    {
        free(h_n);
        free(h_link);
        cudaFree(d_link);
        cudaFree(d_n);
        cudaFree(d_state);
    }
    void set_d_n(int n)
    {
        assert(n <= n_max);
        cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    }
    int get_d_n()
    {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(n <= n_max);
        return n;
    }
    void reset(Check_link check = every_link)
    {
        copy_to_host();
        for (auto i = 0; i < n_max; i++) {
            if (!check(h_link[i].a, h_link[i].b)) continue;

            h_link[i].a = 0;
            h_link[i].b = 0;
        }
        copy_to_device();
    }
    void copy_to_device()
    {
        assert(*h_n <= n_max);
        cudaMemcpy(
            d_link, h_link, n_max * sizeof(Link), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void copy_to_host()
    {
        cudaMemcpy(
            h_link, d_link, n_max * sizeof(Link), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_n <= n_max);
    }
};


template<typename Pt>
using Link_force = void(const Pt* __restrict__ d_X, const int a, const int b,
    const float strength, Pt* d_dX);

template<typename Pt>
__device__ void linear_force(const Pt* __restrict__ d_X, const int a,
    const int b, const float strength, Pt* d_dX)
{
    auto r = d_X[a] - d_X[b];
    auto dist = norm3df(r.x, r.y, r.z);

    atomicAdd(&d_dX[a].x, -strength * r.x / dist);
    atomicAdd(&d_dX[a].y, -strength * r.y / dist);
    atomicAdd(&d_dX[a].z, -strength * r.z / dist);
    atomicAdd(&d_dX[b].x, strength * r.x / dist);
    atomicAdd(&d_dX[b].y, strength * r.y / dist);
    atomicAdd(&d_dX[b].z, strength * r.z / dist);
}

template<typename Pt, Link_force<Pt> force>
__global__ void link(const Pt* __restrict__ d_X, Pt* d_dX,
    const Link* __restrict__ d_link, int n_max, float strength)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_max) return;

    auto a = d_link[i].a;
    auto b = d_link[i].b;
    if (a == b) return;

    force(d_X, a, b, strength, d_dX);
}

// First template is required to work around bug in CUDA 9.2 and 10.
template<typename Pt>
void link_forces(Links& links, const Pt* __restrict__ d_X, Pt* d_dX)
{
    link<Pt, linear_force<Pt>><<<(links.get_d_n() + 32 - 1) / 32, 32>>>(
        d_X, d_dX, links.d_link, links.get_d_n(), links.strength);
}

template<typename Pt, Link_force<Pt> force>
void link_forces(Links& links, const Pt* __restrict__ d_X, Pt* d_dX)
{
    link<Pt, force><<<(links.get_d_n() + 32 - 1) / 32, 32>>>(
        d_X, d_dX, links.d_link, links.get_d_n(), links.strength);
}

// Implementation of solid walls that restrict cell movement as
// Walls are defined as planes normal to a certain direction (e.g. Z axis)
// and their position along that direction is tracked with a "wall node"
// Pairwise forces between cells and the wall node depends on the point to plane
// distance, instead of point to point distance as in cell-cell interactions.
// Other scenarios can be implemented with walls oriented in different
// directions, or with multiple walls.

// Pairwise forces between cells and the wall.
template<typename Pt>
using Wall_force = void(const Pt* __restrict__ d_X, const int i,
    const int wall_idx, Pt* d_dX, int* d_nints, const float w_off_s);

// Wall force implementation with one wall normal to Z axis
template<typename Pt>
__device__ void xy_wall_relu_force(const Pt* __restrict__ d_X, const int i,
    const int wall_idx, Pt* d_dX, int* d_nints)
{
    auto Xwall = d_X[wall_idx].z;
    auto dist_wall = fabs(d_X[i].z - Xwall);
    if (dist_wall < 1.0f) {
        auto F = fmaxf(0.8 - dist_wall, 0) - fmaxf(dist_wall - 0.8, 0);
        d_dX[i].z += F;

        atomicAdd(&d_dX[wall_idx].z, -F);
        atomicAdd(&d_nints[wall_idx], 1);
    }
}

template<typename Pt, Wall_force<Pt> force>
__global__ void wall(const Pt* __restrict__ d_X, Pt* d_dX, int n_max,
    int wall_idx, int* d_nints, const float w_off_s)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_max) return;
    if (i == wall_idx) return;

    force(d_X, i, wall_idx, d_dX, d_nints, w_off_s);
}

template<typename Pt>
__global__ void update_wall_node(
    Pt* d_dX, int n_max, int wall_idx, int* d_nints)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_max) return;

    if (d_nints[i] > 0) {
        d_dX[i].x *= 1 / float(d_nints[i]);
        d_dX[i].y *= 1 / float(d_nints[i]);
        d_dX[i].z *= 1 / float(d_nints[i]);
    }
}

// Use this when there is wall node, but no links
template<typename Pt, Wall_force<Pt> force>
void wall_forces(
    const int n, const Pt* __restrict__ d_X, Pt* d_dX, const int wall_idx)
{
    int* d_nints;
    cudaMalloc(&d_nints, 2 * sizeof(int));
    thrust::fill(thrust::device, d_nints, d_nints + 2, 0);

    wall<Pt, force><<<(n + 32 - 1) / 32, 32>>>(d_X, d_dX, n, wall_idx, d_nints);

    update_wall_node<<<1, 1>>>(d_dX, n, wall_idx, d_nints);
}

// Use this instead of "link_forces" when there is wall node and links
template<typename Pt, Link_force<Pt> l_force, Wall_force<Pt> w_force>
void link_wall_forces(Links& links, const int n, const Pt* __restrict__ d_X,
    Pt* d_dX, const int wall_idx)
{
    link<Pt, l_force><<<(links.get_d_n() + 32 - 1) / 32, 32>>>(
        d_X, d_dX, links.d_link, links.get_d_n(), links.strength);

    int* d_nints;
    cudaMalloc(&d_nints, 1 * sizeof(int));
    thrust::fill(thrust::device, d_nints, d_nints + 1, 0);

    wall<Pt, w_force>
        <<<(n + 32 - 1) / 32, 32>>>(d_X, d_dX, n, wall_idx, d_nints);

    update_wall_node<<<1, 1>>>(d_dX, n, wall_idx, d_nints);
}

// Wall force implementation with one wall normal to Z axis
template<typename Pt>
__device__ void boundary_force(const Pt* __restrict__ d_X, const int i,
    const int wall_idx, Pt* d_dX, int* d_nints)
{
    auto Xwall = -2;
    auto dist_wall = fabs(d_X[i].x - Xwall);
    if (dist_wall < 0.5f) {
        auto F = fmaxf(2.0 - dist_wall, 0) - fmaxf(dist_wall - 2.0, 0);
        d_dX[i].x += F;
    }
}

// wall force for multiple walls
template<typename Pt, Wall_force<Pt> force>
void wall_forces_mult(const int n, const Pt* __restrict__ d_X, Pt* d_dX,
    const int wall_idx, const float w_off_s)
{
    int* d_nints;
    cudaMalloc(&d_nints, 2 * sizeof(int));
    thrust::fill(thrust::device, d_nints, d_nints + 2, 0);

    // Launch a single kernel that handles multiple walls
    wall<Pt, force>
        <<<(n + 32 - 1) / 32, 32>>>(d_X, d_dX, n, wall_idx, d_nints, w_off_s);
}


// Compute perpendicular distance to arbitrary plane
template<typename Pt>
__device__ float distance_to_plane(
    const Pt& point, const float* norm, const float off)
{  // the distance will be negative if the point is on the opposite side of the
   // normal from the origin
    return -(norm[0] * point.x + norm[1] * point.y + norm[2] * point.z - off) /
           norm3df(norm[0], norm[1], norm[2]);
}

template<typename Pt>
__device__ void boundary_forces_mult(const Pt* __restrict__ d_X, const int i,
    const int wall_idx, Pt* d_dX, int* d_nints, const float w_off_s)
{
    // Walls are defined as Ax + By + Cz = D with normals A, B, C and offset D
    float w_norms[5][3] = {
        {-1, 0, 0},  // plane at x=-off   (off short for offset)
        {1, 0, 0},   // plane at x=off
        {0, 1, 0},   // plane at y=off
        {0, -1, 0},  // plane at y=-off
    };

    float w_off[] = {3.75, 3.75, 1.25, 1.25};  // Offsets, order same as w_norms
    // float w_off_s = 1;                 // Scaling factor for the wall offsets
    int num_walls = 4;   // Number of walls
    float w_fmax = 0;    // Force from wall at the wall
    float w_thresh = 0;  // Threshold distance for applying force

    for (int w = 0; w < num_walls; ++w) {
        float dist_wall =  // Get the distance to the wall (wth wall index)
            distance_to_plane(d_X[i], w_norms[w], w_off[w] * w_off_s);
        if (dist_wall < w_thresh) {
            auto F_mag = fmaxf(w_fmax - dist_wall, 0);  // linear force increase
            // Compute the force direction (normalized wall normal)
            float norm = sqrtf(w_norms[w][0] * w_norms[w][0] +
                               w_norms[w][1] * w_norms[w][1] +
                               w_norms[w][2] * w_norms[w][2]);
            float F_dir_x =
                -w_norms[w][0] /
                norm;  // Direction of the force is opposite to the wall normal
                       // (the vector from the origin to the wall)
            float F_dir_y = -w_norms[w][1] / norm;
            // float F_dir_z = w_norms[w][2] / norm;

            // Apply the force to the cell
            d_dX[i].x += F_mag * F_dir_x;
            d_dX[i].y += F_mag * F_dir_y;
            // d_dX[i].z += F_mag * F_dir_z;
            // printf("F_mag: %f, F_dir_x: %f, F_mag * F_dir_x: %f\n", F_mag,
            // F_dir_x, F_mag * F_dir_x);
        }
    }
}
