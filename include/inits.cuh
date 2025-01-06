// Initial states
#pragma once

#include <assert.h>
#include <time.h>
#include <iostream>
#include <random>


template<typename Pt, template<typename> class Solver>
class Solution;


template<typename Pt, template<typename> class Solver>
void random_disk(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    std::random_device rd;
    srand(rd());
    // Radius based on hexagonal lattice
    auto r_max = pow((*points.h_n - n_0) / 0.9069, 1. / 2) * dist_to_nb / 2;
    for (auto i = n_0; i < *points.h_n; i++) {
        auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 2);
        auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;
        points.h_X[i].x = 0;
        points.h_X[i].y = r * sin(phi);
        points.h_X[i].z = r * cos(phi);
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void random_disk_z(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    std::random_device rd;
    srand(rd());
    // Radius based on hexagonal lattice
    auto r_max = pow((*points.h_n - n_0) / 0.9069, 1. / 2) * dist_to_nb / 2;
    for (auto i = n_0; i < *points.h_n; i++) {
        auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 2);
        auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;
        points.h_X[i].x = r * sin(phi);
        points.h_X[i].y = r * cos(phi);
        points.h_X[i].z = 0;
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void two_random_disk_z(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    std::random_device rd;
    srand(rd());
    // Radius based on hexagonal lattice
    auto r_max = pow((*points.h_n - n_0) / 0.9069, 1. / 2) * dist_to_nb / 2;
    for (auto i = n_0; i < *points.h_n; i++) {
        auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 2);
        auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;

        // define two centers, one for the first half of the initial cells at x=-0.5 and one for rest at x=0.5
        float cx = (i < *points.h_n / 2) ? -dist_to_nb * 20 : dist_to_nb * 20;

        points.h_X[i].x = cx + r * sin(phi);
        points.h_X[i].y = r * cos(phi);
        points.h_X[i].z = 0;
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void volk_zebra_2D(
    float A_dist, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    //std::random_device rd;
    //srand(rd());
    // define 5 stripes
    std::vector<float> y_pos{0.0,0.418,0.5,0.582,1.0}; //initial positions of cell stripes

    // int cellsPer_Astripe = 2 / A_dist;
    //int cellsPer_Bstripe = 2 / B_dist;
    int cellsPer_stripe = *points.h_n / y_pos.size(); // calculate no. cells in each stripe
    for (int i = 0; i < y_pos.size(); i++) {
        // int cellsPer_stripe = *points.h_n /
        for (int j = 0; j < cellsPer_stripe; j++) {
            int index = i * cellsPer_stripe + j;
            if (y_pos[i] == 0.5) points.h_X[index].x = j * 0.05; //initialise xanthophores closer together
            else points.h_X[index].x = j * 0.1;
            // points.h_X[index].x = j * dist_to_nb;
            points.h_X[index].y = y_pos[i];
            points.h_X[index].z = 0;
            // if (i == 2) {
            //     //points.h_X[index].cell_type = 1; // only cells in the middle stripe are iridophores
            //     d_cell_type[index] = 1;
            // } else {
            //     //points.h_X[index].cell_type = 2;
            //     d_cell_type[index] = 2;
            // }
        }
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void volk_zebra_3stripe(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    // define 7 stripes
    std::vector<float> y_pos{0.0,0.209,0.418,0.5,0.582,0.791,1.0}; //initial positions of cell stripes

    int cellsPer_stripe = *points.h_n / y_pos.size(); // calculate no. cells in each stripe
    for (int i = 0; i < y_pos.size(); i++) {
        for (int j = 0; j < cellsPer_stripe; j++) {
            int index = i * cellsPer_stripe + j;
            points.h_X[index].x = j * dist_to_nb;
            points.h_X[index].y = y_pos[i];
            points.h_X[index].z = 0;
        }
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void volk_zebra_3stripe_ablated_empty(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    // define 7 stripes
    std::vector<float> y_pos{0.0,0.209,0.418,0.5,0.582,0.791,1.0}; //initial positions of cell stripes
    int gap_size = 10; // define the size of the gap in the tissue in number of cells 

    int cellsPer_stripe = *points.h_n / y_pos.size(); // calculate no. cells in each stripe
    for (int i = 0; i < y_pos.size(); i++) {
        for (int j = 0; j < cellsPer_stripe; j++) {
            int index = i * cellsPer_stripe + j;
            if (j < cellsPer_stripe / 2) {
                points.h_X[index].x = j * dist_to_nb;
            } else {
                points.h_X[index].x = (j * dist_to_nb) + (gap_size * dist_to_nb);
            }
            
            points.h_X[index].y = y_pos[i];
            points.h_X[index].z = 0;
        }
    }
    points.copy_to_device();
}

// template<typename Pt, template<typename> class Solver>
// void volk_zebra_3stripe_ablated(
//     float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
// {
//     assert(n_0 < *points.h_n);
//     // define 7 stripes
//     std::vector<float> y_pos{0.0,0.209,0.418,0.5,0.582,0.791,1.0}; //initial positions of cell stripes

//     int cellsPer_stripe = *points.h_n / y_pos.size(); // calculate no. cells in each stripe
//     for (int i = 0; i < y_pos.size(); i++) {
//         for (int j = 0; j < cellsPer_stripe; j++) {
//             int index = i * cellsPer_stripe + j;
//             if (j < cellsPer_stripe * 0.4) {
//                 points.h_X[index].x = j * dist_to_nb;
//                 points.h_X[index].y = y_pos[i];
//                 }
//             if (j >= cellsPer_stripe * 0.4 and
//                 j < cellsPer_stripe * 0.6
//             ) {
//                 float xl = cellsPer_stripe * dist_to_nb * 0.4;
//                 float xu = cellsPer_stripe * dist_to_nb * 0.6;
//                 float yl = 0;
//                 float yu = 1;
//                 points.h_X[index].x = (rand() / (RAND_MAX + 1.)) * (xu - xl);
//                 points.h_X[index].y = (rand() / (RAND_MAX + 1.)) * (yu - yl);
//             }
//             if (j >= cellsPer_stripe * 0.6) {
//                 points.h_X[index].x = (j * dist_to_nb) + (cellsPer_stripe * dist_to_nb * 0.6);
//                 points.h_X[index].y = y_pos[i];
//             }

            
            
//             points.h_X[index].z = 0;
//         }
//     }
//     points.copy_to_device();
// }

template<typename Pt, template<typename> class Solver>
void volk_zebra_3stripe_ablated(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    // define 7 stripes
    std::vector<float> y_pos{0.0,0.209,0.418,0.5,0.582,0.791,1.0};  //initial positions of cell stripes
    int gap_size = 10;                                              // define the size of the gap in the tissue in number of cells 
    int nc_gap = pow(gap_size, 2);                                  // number of cells to fill the gap
    int nc_stripe = *points.h_n - nc_gap;                         // total number of cells in all stripes (just not the gap)
    // printf("nc_stripe %d\n", nc_stripe);

    int cellsPer_stripe = (nc_stripe / y_pos.size()); // calculate no. cells in each stripe
    // printf("cellsPer_stripe: %d\n", cellsPer_stripe);
    // printf("*points.h_n: %d\n", *points.h_n);
    // save gap_size^2 number of cells to fill the gap
    for (int i = 0; i < y_pos.size(); i++) {
        for (int j = 0; j < cellsPer_stripe; j++) {
            int index = i * cellsPer_stripe + j;
            if (j < cellsPer_stripe / 2) {
                points.h_X[index].x = j * dist_to_nb;
            } else {
                points.h_X[index].x = (j * dist_to_nb) + (gap_size * dist_to_nb);
            }
            points.h_X[index].y = y_pos[i];
            points.h_X[index].z = 0;
            // printf("index %d\n", index);
            // printf("pos: x = %f, y = %f, z = %f\n", points.h_X[index].x, points.h_X[index].y, points.h_X[index].z);
        }
    }

    // initialise randomness
    std::random_device rd;
    srand(rd());
    // fill gap with a random distribution of cells
    for (int index = nc_stripe; index < *points.h_n; index++) {
        float xl = (cellsPer_stripe / 2) * dist_to_nb;
        float xu = ((cellsPer_stripe / 2) * dist_to_nb) + (gap_size * dist_to_nb);
        float yl = 0;
        float yu = 1;
        points.h_X[index].x = xl + ((xu - xl) * rand() / (RAND_MAX + 1.));
        points.h_X[index].y = yl + ((yu - yl) * rand() / (RAND_MAX + 1.));
        points.h_X[index].z = 0;  
        // printf("index %d\n", index);
        // printf("pos: x = %f, y = %f, z = %f\n", points.h_X[index].x, points.h_X[index].y, points.h_X[index].z);
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void random_sphere(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);
    std::random_device rd;
    srand(rd());
    // Radius based on random sphere packing
    auto r_max = pow((*points.h_n - n_0) / 0.64, 1. / 3) * dist_to_nb / 2;
    for (auto i = n_0; i < *points.h_n; i++) {
        auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 3);
        auto theta = acos(2. * rand() / (RAND_MAX + 1.) - 1);
        auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;
        points.h_X[i].x = r * sin(theta) * cos(phi);
        points.h_X[i].y = r * sin(theta) * sin(phi);
        points.h_X[i].z = r * cos(theta);
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void random_cuboid(float dist_to_nb, float3 minimum, float3 maximum,
    Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);

    auto dimension = maximum - minimum;
    auto cube_volume = dimension.x * dimension.y * dimension.z;
    auto sphere_volume = 4. / 3 * M_PI * pow(dist_to_nb / 2, 3);
    auto n = cube_volume / sphere_volume * 0.64;  // Sphere packing

    assert(n_0 + n < *points.h_n);
    *points.h_n = n_0 + n;

    std::random_device rd;
    srand(rd());
    for (auto i = n_0; i < *points.h_n; i++) {
        points.h_X[i].x = minimum.x + dimension.x * (rand() / (RAND_MAX + 1.));
        points.h_X[i].y = minimum.y + dimension.y * (rand() / (RAND_MAX + 1.));
        points.h_X[i].z = minimum.z + dimension.z * (rand() / (RAND_MAX + 1.));
    }
    points.copy_to_device();
}


template<typename Pt>
__device__ Pt relu_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    if (i == j) return dF;

    if (dist > 1.f) return dF;

    auto F = fmaxf(0.8f - dist, 0) * 2.f - fmaxf(dist - 0.8f, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    return dF;
}

template<typename Pt, template<typename> class Solver>
void relaxed_sphere(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    random_sphere(0.6, points, n_0);

    int relax_steps;
    if (*points.h_n <= 100)
        relax_steps = 500;
    else if (*points.h_n <= 1000)
        relax_steps = 1000;
    else if (*points.h_n <= 6000)
        relax_steps = 2000;
    else
        relax_steps = 3000;
    if (*points.h_n > 10000)
        std::cout << "Warning: The system is quite large, it may "
                  << "not be completely relaxed." << std::endl;

    for (auto i = 0; i < relax_steps; i++)
        points.template take_step<relu_force>(0.1f);
    points.copy_to_host();

    auto scale = dist_to_nb / 0.8;
    for (auto i = 0; i < *points.h_n; i++) {
        points.h_X[i].x *= scale;
        points.h_X[i].y *= scale;
        points.h_X[i].z *= scale;
    }
    points.copy_to_device();
}

template<typename Pt, template<typename> class Solver>
void relaxed_cuboid(float dist_to_nb, float3 minimum, float3 maximum,
    Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    auto scale = dist_to_nb / 0.8;
    random_cuboid(0.8, minimum / scale, maximum / scale, points, n_0);

    int relax_steps;
    if (*points.h_n <= 3000)
        relax_steps = 1000;
    else if (*points.h_n <= 12000)
        relax_steps = 2000;
    else
        relax_steps = 3000;
    if (*points.h_n > 15000)
        std::cout << "Warning: The system is quite large, it may "
                  << "not be completely relaxed." << std::endl;

    for (int i = 0; i < relax_steps; i++)
        points.template take_step<relu_force>(0.1f);
    points.copy_to_host();

    for (auto i = 0; i < *points.h_n; i++) {
        points.h_X[i].x *= scale;
        points.h_X[i].y *= scale;
        points.h_X[i].z *= scale;
    }
    points.copy_to_device();
}


template<typename Pt, template<typename> class Solver>
void regular_hexagon(
    float dist_to_nb, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);

    auto beta = M_PI / 3.f;

    // Point in center
    auto cell_counter = n_0;
    points.h_X[cell_counter].x = 0.f;
    points.h_X[cell_counter].y = 0.f;
    points.h_X[cell_counter].z = 0.f;
    cell_counter++;
    if (cell_counter == *points.h_n) {
        points.copy_to_device();
        return;
    }
    auto i = 1;
    while (true) {
        for (auto j = 0; j < 6; j++) {
            // Main axis point
            auto angle = beta * j;
            float3 p{-dist_to_nb * i * sinf(angle),
                dist_to_nb * i * cosf(angle), 0.f};
            points.h_X[cell_counter].x = p.x;
            points.h_X[cell_counter].y = p.y;
            points.h_X[cell_counter].z = p.z;
            cell_counter++;
            if (cell_counter == *points.h_n) {
                points.copy_to_device();
                return;
            }
            // Intermediate points
            auto n_int = i - 1;
            if (n_int < 1) continue;
            auto next_angle = beta * (j + 1);
            float3 q{-dist_to_nb * i * sinf(next_angle),
                dist_to_nb * i * cosf(next_angle), 0.f};
            auto v = q - p;
            auto modulus = sqrt(pow(v.x, 2) + pow(v.y, 2));
            v = v * (1.f / modulus);
            for (auto k = 1; k <= n_int; k++) {
                auto u = v * modulus * (float(k) / float(n_int + 1));
                points.h_X[cell_counter].x = p.x + u.x;
                points.h_X[cell_counter].y = p.y + u.y;
                points.h_X[cell_counter].z = p.z + u.z;
                cell_counter++;
                if (cell_counter == *points.h_n) {
                    points.copy_to_device();
                    return;
                }
            }
        }
        i++;
    }
}

template<typename Pt, template<typename> class Solver>
void regular_rectangle(
    float dist_to_nb, int nx, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{
    assert(n_0 < *points.h_n);

    auto cell_counter = n_0;
    if (cell_counter == *points.h_n) {
        points.copy_to_device();
        return;
    }
    auto i = 0;
    auto full = false;
    while (!full) {
        float py =  i * sqrt(pow(dist_to_nb, 2) - pow(dist_to_nb/2.f, 2));
        float row_offset = 0.0;
        if(i%2 != 0)
            row_offset = dist_to_nb/2.f;
        for (auto j = 0; j < nx; j++) {
            points.h_X[cell_counter].x = row_offset + j * dist_to_nb;
            points.h_X[cell_counter].y = py;
            points.h_X[cell_counter].z = 0.0f;
            cell_counter++;
            if (cell_counter == *points.h_n) {
                points.copy_to_device();
                full = true;
                return;
            }
        }
        i++;
    }
}

template<typename Pt, template<typename> class Solver>
void regular_rectangle_w_spot(
    int sp_size, float dist_to_nb, int nx, Solution<Pt, Solver>& points, unsigned int n_0 = 0)
{   // sp_size gives the number of cells in the initial iridophore aggregation
    assert(n_0 < *points.h_n);

    auto cell_counter = n_0;
    int nrows = 0;
    if (cell_counter == *points.h_n) {
        points.copy_to_device();
        return;
    }
    // initialise rectangle
    auto i = 0;
    auto full = false;
    while (!full) { // exit while loop when the rectangle is full
        float py =  i * sqrt(pow(dist_to_nb, 2) - pow(dist_to_nb/2.f, 2));
        float row_offset = 0.0;
        if(i%2 != 0)
            row_offset = dist_to_nb/2.f;
        for (auto j = 0; j < nx; j++) {
            points.h_X[cell_counter].x = row_offset + j * dist_to_nb;
            points.h_X[cell_counter].y = py;
            points.h_X[cell_counter].z = 0.0f;
            cell_counter++;
            if (cell_counter == *points.h_n - sp_size) {
                full = true;
                nrows = i + 1; //save the number of rows in the rectangle
                // printf("i: %d\n", i);
                // printf("nrows: %d\n", nrows);
                break;
            }
        }
        i++;
    }
    // add spot cells at the far left of the rectangle, column by column
    auto j = 0;
    full = false;
    while (!full) {
        for (auto i = 0; i < nrows; i++) {

            float row_offset = (i%2 != 0) ? 0 : dist_to_nb / 2.f; // offset to place spot cells between non-spot cells
            float px = row_offset + j * dist_to_nb;
            float py = i * sqrt(pow(dist_to_nb, 2) - pow(dist_to_nb / 2.f, 2));
            points.h_X[cell_counter].x = px;
            points.h_X[cell_counter].y = py;
            points.h_X[cell_counter].z = 0.0f;
            cell_counter++;
            if (cell_counter == *points.h_n) {
                full = true;
                break;
            }
        }
        j++;
    }
    points.copy_to_device();
}

