#pragma once

#include <math.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dtypes.cuh"
#include "solvers.cuh"
#include "utils.cuh"

struct Triangle_d {
    float3 V0, V1, V2;
    float3 n;  // normal (optional, for intersection)
    void calculate_normal()
    {
        auto v = V2 - V0;
        auto u = V1 - V0;
        n = float3{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x};
        n /= sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    }
};


struct Mesh_d {
    Triangle_d* d_facets;
    int n_facets;
};

__host__ __device__ bool intersect_ray_triangle(
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
__host__ __device__ bool intersect(
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
__host__ __device__ bool test_exclusion(const Mesh_d& mesh, const Pt d_X)
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
    // printf("ni%d\n", n_intersections);
    // Even: outside, Odd: inside
    return (n_intersections % 2 == 0);
}

// From Ericson, C., 2004. Real-Time Collision Detection, 0 ed. CRC Press.
// https://doi.org/10.1201/b14581 section 5.1.5
template<typename Pt>
__host__ __device__ float3 closest_point_on_triangle(
    const Pt d_X, const Triangle_d& tri)
{
    // find closest point on triangle abc to point p
    float3 p = make_float3(d_X.x, d_X.y, d_X.z);
    float3 a = tri.V0, b = tri.V1, c = tri.V2;
    // Check if P in vertex region ouside A
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot_product(ab, ap);
    float d2 = dot_product(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;  // Barycentric coords (1,0,0)

    // Check if P in vertex region outside B
    float3 bp = p - b;
    float d3 = dot_product(ab, bp);
    float d4 = dot_product(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;  // Barycentric coords (0,1,0)

    // Check if P in edge region of AB, if so return projection of P onto AB
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + ab * v;  // Barycentric coords (1-v,v,0)
    }

    // Check if P in vertex region outside C
    float3 cp = p - c;
    float d5 = dot_product(ab, cp);
    float d6 = dot_product(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;  // Barycentric coords (0,0,1)

    // Check if P in edge region of AC, if so return projection of P onto AC
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + ac * w;  // Barycentric coords (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;  // Barycentric coords (0,1-w,w)
    }

    // P inside face region. Compute Q through its barycentric coordinates
    // (u,v,w)
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v +
           ac * w;  // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
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
            tri.calculate_normal();
            facets.push_back(tri);
        }
    }
    return facets;
}

class Fin {
public:
    std::vector<Triangle_d> h_facets;
    Mesh_d mesh;
    int n_facets;
    Fin(std::string file_name, std::string out_name);
    void grow(float amount);
    void write_vtk();
    std::string out_name;
    int time_step = 0;
    float3 get_maximum();
    float3 get_minimum();
};

Fin::Fin(std::string file_name, std::string out_name = "NONE")
{
    h_facets = read_facets_from_vtk(file_name);
    n_facets = h_facets.size();
    mesh.n_facets = n_facets;
    cudaMalloc(&mesh.d_facets, n_facets * sizeof(Triangle_d));
    cudaMemcpy(mesh.d_facets, h_facets.data(), n_facets * sizeof(Triangle_d),
        cudaMemcpyHostToDevice);
    this->out_name = out_name;
}

void Fin::grow(float amount)
{
    for (int i = 0; i < n_facets; ++i) {  // don't scale in z
        h_facets[i].V0.x *= amount;
        h_facets[i].V0.y *= amount;
        h_facets[i].V1.x *= amount;
        h_facets[i].V1.y *= amount;
        h_facets[i].V2.x *= amount;
        h_facets[i].V2.y *= amount;
        // z remains unchanged
        float3 edge1 = h_facets[i].V1 - h_facets[i].V0;
        float3 edge2 = h_facets[i].V2 - h_facets[i].V0;
        h_facets[i].calculate_normal();
    }
    cudaMemcpy(mesh.d_facets, h_facets.data(), n_facets * sizeof(Triangle_d),
        cudaMemcpyHostToDevice);
}

void Fin::write_vtk()
{
    std::string current_path =
        "output/" + out_name + "_" + std::to_string(time_step) + ".vtk";
    std::ofstream file(current_path);
    file << "# vtk DataFile Version 3.0\n";
    file << "Fin mesh\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";
    file << "POINTS " << n_facets * 3 << " float\n";
    for (const auto& tri : h_facets) {
        file << tri.V0.x << " " << tri.V0.y << " " << tri.V0.z << "\n";
        file << tri.V1.x << " " << tri.V1.y << " " << tri.V1.z << "\n";
        file << tri.V2.x << " " << tri.V2.y << " " << tri.V2.z << "\n";
    }
    file << "POLYGONS " << n_facets << " " << n_facets * 4 << "\n";
    for (int i = 0; i < n_facets; ++i) {
        file << "3 " << i * 3 << " " << i * 3 + 1 << " " << i * 3 + 2 << "\n";
    }
    file.close();
    time_step++;
}


float3 Fin::get_maximum()
{
    float3 max_pt = make_float3(-1e30f, -1e30f, -1e30f);
    for (const auto& tri : h_facets) {
        max_pt.x = fmaxf(max_pt.x, fmaxf(tri.V0.x, fmaxf(tri.V1.x, tri.V2.x)));
        max_pt.y = fmaxf(max_pt.y, fmaxf(tri.V0.y, fmaxf(tri.V1.y, tri.V2.y)));
        max_pt.z = fmaxf(max_pt.z, fmaxf(tri.V0.z, fmaxf(tri.V1.z, tri.V2.z)));
    }
    return max_pt;
}

float3 Fin::get_minimum()
{
    float3 min_pt = make_float3(1e30f, 1e30f, 1e30f);
    for (const auto& tri : h_facets) {
        min_pt.x = fminf(min_pt.x, fminf(tri.V0.x, fminf(tri.V1.x, tri.V2.x)));
        min_pt.y = fminf(min_pt.y, fminf(tri.V0.y, fminf(tri.V1.y, tri.V2.y)));
        min_pt.z = fminf(min_pt.z, fminf(tri.V0.z, fminf(tri.V1.z, tri.V2.z)));
    }
    return min_pt;
}