#include "../msh_utils/mesh_data.hpp"

#include <Mathematics/Vector3.h>
#include <Mathematics/Triangle.h>
#include <Mathematics/DistPointTriangle.h>

#include <vector>
#include <cmath>

static gte::Vector3<double> toGte(const Vec3& p)
{
    return {p.x, p.y, p.z};
}

static Vec3 sub(const Vec3& a, const Vec3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vec3 cross(const Vec3& a, const Vec3& b)
{
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

static double dot(const Vec3& a, const Vec3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static bool rayIntersectsTriangle(const Vec3& p,
                                  const Vec3& a,
                                  const Vec3& b,
                                  const Vec3& c)
{
    const double EPS = 1e-9;

    Vec3 dir{1.0, 0.123, 0.047};

    Vec3 e1 = sub(b, a);
    Vec3 e2 = sub(c, a);

    Vec3 h = cross(dir, e2);
    double det = dot(e1, h);

    if (std::fabs(det) < EPS)
        return false;

    double invDet = 1.0 / det;

    Vec3 s = sub(p, a);
    double u = invDet * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    Vec3 q = cross(s, e1);
    double v = invDet * dot(dir, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    double t = invDet * dot(e2, q);

    return t > EPS;
}

static bool pointInsideMesh(const Vec3& p, const MeshData& mesh)
{
    int count = 0;

    for (const auto& tri : mesh.triangles) {
        const Vec3& a = mesh.nodes[tri.v1];
        const Vec3& b = mesh.nodes[tri.v2];
        const Vec3& c = mesh.nodes[tri.v3];

        if (rayIntersectsTriangle(p, a, b, c))
            count++;
    }

    return count % 2 == 1;
}

static bool nearTaggedSurface(const Vec3& p,
                              const std::vector<BoundaryTriangle>& triangles,
                              const MeshData& mesh,
                              double threshold)
{
    const double threshold2 = threshold * threshold;

    gte::DCPPoint3Triangle3<double> query;
    gte::Vector3<double> point = toGte(p);

    for (const auto& tri : triangles) {
        const Vec3& a = mesh.nodes[tri.v1];
        const Vec3& b = mesh.nodes[tri.v2];
        const Vec3& c = mesh.nodes[tri.v3];

        gte::Triangle3<double> triangle{
            {
                toGte(a),
                toGte(b),
                toGte(c)
            }
        };

        auto result = query(point, triangle);

        if (result.sqrDistance <= threshold2)
            return true;
    }

    return false;
}