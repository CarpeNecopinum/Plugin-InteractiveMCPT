#pragma once

#include <cuda.h>
#include "cutil_math.h"
#include <curand_kernel.h>
#include "../MonteCuda.hh"
#include "KdTree.hh"

struct mcIntersection
{
    float3 position;
    float3 normal;
    float depth;
    mcMaterial material;
};

__device__ float3 mcMirror(const float3& direction, const float3& normal);
__device__ float3 clampToAxis(const float3 &n);
__device__ bool mcBaryTest(float3 point, mcTriangle triangle);
__device__ mcIntersection mcIntersectTriangle(mcRay ray, mcTriangle triangle);
__device__ mcIntersection mcIntersectScene(mcRay ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount);
__device__ float3 mcTriangleNormal(mcTriangle tri);

__device__ float3 mcMirror(const float3& direction, const float3& normal)
{
    return direction - normal * 2.f * dot(normal, direction);
}


__device__ float3 clampToAxis(const float3 &n) {
    if(abs(n.x) > std::fabs(n.y) && std::fabs(n.x) > std::fabs(n.z)) {
        return make_float3(0,0,1);
    } else if(std::fabs(n.y) > std::fabs(n.x) && std::fabs(n.y) > std::fabs(n.z)) {
        return make_float3(1,0,0);
    }
    return make_float3(0,1,0);
}
__device__ bool mcBaryTest(float3 point, mcTriangle triangle)
{
    float3 p = point;
    float3 u = triangle.corners[0];
    float3 v = triangle.corners[1];
    float3 w = triangle.corners[2];

    float3 result;

    float3  vu = v - u,
    wu = w - u,
    pu = p - u;

    // find largest absolute coordinate of normal
    float nx = vu.y * wu.z - vu.z * wu.y,
                 ny        = vu.z * wu.x - vu.x * wu.z,
                 nz        = vu.x * wu.y - vu.y * wu.x,
                 ax        = abs(nx),
                 ay        = abs(ny),
                 az        = abs(nz);

    unsigned char max_coord;

    if ( ax > ay ) {
        if ( ax > az ) {
            max_coord = 0;
        }
        else {
            max_coord = 2;
        }
    }
    else {
        if ( ay > az ) {
            max_coord = 1;
        }
        else {
            max_coord = 2;
        }
    }

    // solve 2D problem
    switch (max_coord)
    {
        case 0:
            {
                if (1.0+ax == 1.0) return false;
                result.y = 1.0 + (pu.y*wu.z-pu.z*wu.y)/nx - 1.0;
                result.z = 1.0 + (vu.y*pu.z-vu.z*pu.y)/nx - 1.0;
                result.x = 1.0 - result.y - result.z;
            }
            break;

        case 1:
            {
                if (1.0+ay == 1.0) return false;
                result.y = 1.0 + (pu.z*wu.x-pu.x*wu.z)/ny - 1.0;
                result.z = 1.0 + (vu.z*pu.x-vu.x*pu.z)/ny - 1.0;
                result.x = 1.0 - result.y - result.z;
            }
            break;

        case 2:
            {
                if (1.0+az == 1.0) return false;
                result.y = 1.0 + (pu.x*wu.y-pu.y*wu.x)/nz - 1.0;
                result.z = 1.0 + (vu.x*pu.y-vu.y*pu.x)/nz - 1.0;
                result.x = 1.0 - result.y - result.z;
            }
            break;
    }

    return (result.x >= 0.f && result.y >= 0.f && result.z >= 0.f);
}

__device__ mcIntersection mcIntersectTriangle(mcRay ray, mcTriangle triangle)
{
    mcIntersection result;
    result.depth = FLT_MAX;

    result.normal = normalize(cross(triangle.corners[1] - triangle.corners[0], triangle.corners[2] - triangle.corners[0]));
    if (dot(result.normal, ray.direction) > 0.0) return result;

    float depth = dot(triangle.corners[0] - ray.origin, result.normal) / dot(ray.direction, result.normal);
    if (depth <= 0.001f) { return result; }

    float3 intersection = ray.origin + depth * ray.direction;
    result.position = intersection;

    if (mcBaryTest(intersection, triangle)) result.depth = depth;

    return result;
}

__device__ bool mcIntersectAABB(mcRay ray, float3 minCorner, float3 maxCorner)
{
    float t1 = (minCorner.x - ray.origin.x) / ray.direction.x;
    float t2 = (maxCorner.x - ray.origin.x) / ray.direction.x;
    float t3 = (minCorner.y - ray.origin.y) / ray.direction.y;
    float t4 = (maxCorner.y - ray.origin.y) / ray.direction.y;
    float t5 = (minCorner.z - ray.origin.z) / ray.direction.z;
    float t6 = (maxCorner.z - ray.origin.z) / ray.direction.z;

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    if ((tmax < 0) || (tmin >= tmax)) return false;
    return true;
}

template<int LEVEL>
__device__ mcIntersection mcIntersectKdTree(mcRay ray, mcMaterial* mats, KdTree<LEVEL>* tree)
{
    mcIntersection left, right;
    left.depth = FLT_MAX; right.depth = FLT_MAX;

    if (mcIntersectAABB(ray, tree->bbMin, tree->bbMax))
    {
        left = mcIntersectKdTree(ray, mats, &(tree->left));
        right = mcIntersectKdTree(ray, mats, &(tree->right));
        if (left.depth > right.depth) left = right;
    }
    return left;
}

template<>
__device__ mcIntersection mcIntersectKdTree<0>(mcRay ray, mcMaterial* mats, KdTree<0>* tree)
{
    mcIntersection result;
    result.depth = FLT_MAX;

    if (mcIntersectAABB(ray, tree->bbMin, tree->bbMax))
    {
        for (size_t i = 0; i < tree->triangleCount; ++i)
        {
            mcIntersection next = mcIntersectTriangle(ray, tree->triangles[i]);
            if (next.depth < result.depth)
            {
                result = next;
                result.material = mats[tree->triangles[i].matIndex];
            }
        }
    }
    return result;
}

__device__ mcIntersection mcIntersectScene(mcRay ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount)
{
    mcIntersection result;
    result.depth = FLT_MAX;

    for (size_t i = 0; i < triCount; ++i)
    {
        mcIntersection next = mcIntersectTriangle(ray, geometry[i]);
        if (next.depth < result.depth)
        {
            result = next;
            result.material = mats[geometry[i].matIndex];
        }
    }

    return result;
}


__device__ float3 mcTriangleNormal(mcTriangle tri)
{
    float3 u = tri.corners[1] - tri.corners[0];
    float3 v = tri.corners[2] - tri.corners[0];

    return normalize(make_float3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    ));
}


