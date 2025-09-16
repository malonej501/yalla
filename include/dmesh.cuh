#pragma once

#include <math.h>

#include <fstream>
#include <sstream>
#include <vector>

#include "dtypes.cuh"
#include "solvers.cuh"
#include "utils.cuh"

struct Triangle_d {
    float3 V0, V1, V2;
    float3 n;  // normal (optional, for intersection)
};


struct Mesh_d {
    Triangle_d* d_facets;
    int n_facets;
};

__device__ bool intersect_ray_triangle(
    const float3& ray_origin, const float3& ray_end, const Triangle_d& T)
{
    // Möller–Trumbore intersection algorithm
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    // Möller, T., Trumbore, B., 2005. Fast, minimum storage ray/triangle
    // intersection, in: ACM SIGGRAPH 2005 Courses on   - SIGGRAPH ’05.
    // Presented at the ACM SIGGRAPH 2005 Courses, ACM Press, Los Angeles,
    // California, p. 7. https://doi.org/10.1145/1198555.1198746
    const float epsilon = 1e-6f;  // Floating-point precision

    float3 edge1 = T.V1 - T.V0;
    float3 edge2 = T.V2 - T.V0;
    float3 ray_vector = ray_end - ray_origin;
    float3 ray_cross_e2 = cross_product(ray_vector, edge2);
    float det = dot_product(edge1, ray_cross_e2);
    if (det > -epsilon && det < epsilon) return false;  // Parallel
    // printf("o:%f,%f,%f e:%f,%f,%f\n", ray_origin.x, ray_origin.y,
    // ray_origin.z,
    //     ray_end.x, ray_end.y, ray_end.z);

    float inv_det = 1.0f / det;
    float3 s = ray_origin - T.V0;
    float u = inv_det * dot_product(s, ray_cross_e2);  // get barycentric coords
    if ((u < 0.0f && fabs(u) > epsilon) ||
        (u > 1.0f && fabs(u - 1.0f) > epsilon))
        return false;
    float3 s_cross_e1 = cross_product(s, edge1);
    float v = inv_det * dot_product(ray_vector, s_cross_e1);
    if ((v < 0.0f && fabs(v) > epsilon) ||
        (u + v > 1.0f && fabs(u + v - 1.0f) > epsilon))
        return false;

    // Compute t to find where the intersection point is on the line
    float t = inv_det * dot_product(edge2, s_cross_e1);
    // if (t > epsilon) {
    //     return true;  // Intersection
    // } else {
    //     return false;  // Line intersection but not a ray intersection
    // }
    return true;  // Intersection
}

// Theory and algorithm: http://geomalgorithms.com/a06-_intersect-2.html
__device__ bool intersect(
    const float3& ray_origin, const float3& ray_end, Triangle_d T)
{
    // Find intersection point PI
    auto r = dot_product(T.n, T.V0 - ray_origin) /
             dot_product(T.n, ray_end - ray_origin);
    if (r < 0) return false;  // Ray going away

    auto PI = ray_origin + ((ray_end - ray_origin) * r);

    // Check if PI in T
    auto u = T.V1 - T.V0;
    auto v = T.V2 - T.V0;
    auto w = PI - T.V0;
    auto uu = dot_product(u, u);
    auto uv = dot_product(u, v);
    auto vv = dot_product(v, v);
    auto wu = dot_product(w, u);
    auto wv = dot_product(w, v);
    auto denom = uv * uv - uu * vv;

    auto s = (uv * wv - vv * wu) / denom;
    if (s < 0.0 or s > 1.0) return false;

    auto t = (uv * wu - uu * wv) / denom;
    if (t < 0.0 or (s + t) > 1.0) return false;

    return true;
}


template<typename Pt>
__device__ bool test_exclusion(const Mesh_d& mesh, const Pt d_X)
{
    float3 point = make_float3(d_X.x, d_X.y, d_X.z);  // Start at cell
    // Cast a ray in a fixed direction
    // float3 ray_dir = make_float3(0.22788f, 0.38849f, 0.81499f);
    float3 ray_dir = make_float3(1.0f, 1.0f, 1.0f);
    float3 ray_end =
        point + ray_dir * 1e6f;  // Large value to ensure intersection
    // printf("nf%d\n", mesh.n_facets);

    int n_intersections = 0;
    for (int j = 0; j < mesh.n_facets; ++j) {
        if (intersect(point, ray_end, mesh.d_facets[j])) { n_intersections++; }
    }
    printf("ni%d\n", n_intersections);
    // Even: outside, Odd: inside
    return (n_intersections % 2 == 0);
}

std::vector<Triangle_d> read_facets_from_vtk(const std::string& filename)
{
    std::vector<Triangle_d> facets;
    std::ifstream file(filename);
    std::string line;
    bool found_points = false, found_polygons = false;
    std::vector<float3> vertices;

    // Find and read points
    while (std::getline(file, line)) {
        if (line.find("POINTS") != std::string::npos) {
            found_points = true;
            break;
        }
    }
    if (found_points) {
        std::istringstream iss(line);
        std::string tmp;
        int n_points;
        iss >> tmp >> n_points;
        while (vertices.size() < n_points && std::getline(file, line)) {
            std::istringstream iss_points(line);
            float x, y, z;
            while (iss_points >> x >> y >> z) {
                vertices.push_back(make_float3(x, y, z));
            }
        }
    }

    // Find and read polygons
    while (std::getline(file, line)) {
        if (line.find("POLYGONS") != std::string::npos) {
            found_polygons = true;
            break;
        }
    }
    if (found_polygons) {
        std::istringstream iss(line);
        std::string tmp;
        int n_polys;
        iss >> tmp >> n_polys;
        for (int i = 0; i < n_polys; ++i) {
            std::getline(file, line);
            std::istringstream iss_poly(line);
            int n_verts, v0, v1, v2;
            iss_poly >> n_verts >> v0 >> v1 >> v2;
            Triangle_d tri;
            tri.V0 = vertices[v0];
            tri.V1 = vertices[v1];
            tri.V2 = vertices[v2];
            // Compute normal if needed
            float3 edge1 = tri.V1 - tri.V0;
            float3 edge2 = tri.V2 - tri.V0;
            tri.n =
                cross_product(edge1, edge2);  // You may want to normalize this
            facets.push_back(tri);
        }
    }
    return facets;
}
