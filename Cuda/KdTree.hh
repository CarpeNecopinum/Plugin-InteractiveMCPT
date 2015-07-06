#pragma once

#include <vector_types.h>
#include <vector>
#include <algorithm>
#include "../MonteCuda.hh"
#include "Geometry.hh"
#include "cutil_math.h"

inline float getCoord(const float3 vector, size_t index) {
    switch(index)
    {
        case 0: return vector.x;
        case 1: return vector.y;
    }
    return vector.z;
}

template<int AXIS>
inline bool notRightOf(const mcTriangle& tri, float value)
{
    return  (getCoord(tri.corners[0], AXIS) < value)
         || (getCoord(tri.corners[1], AXIS) < value)
         || (getCoord(tri.corners[2], AXIS) < value);
}

template<int AXIS>
inline bool notLeftOf(const mcTriangle& tri, float value)
{
    return  (getCoord(tri.corners[0], AXIS) > value)
         || (getCoord(tri.corners[1], AXIS) > value)
         || (getCoord(tri.corners[2], AXIS) > value);
}

inline float3 triMin(const mcTriangle& tri)
{
    float3 result = tri.corners[0];
    result = fminf(result, tri.corners[1]);
    result = fminf(result, tri.corners[2]);
    return result;
}

inline float3 triMax(const mcTriangle& tri)
{
    float3 result = tri.corners[0];
    result = fmaxf(result, tri.corners[1]);
    result = fmaxf(result, tri.corners[2]);
    return result;
}


template<int LEVELS_LEFT>
struct KdTree
{
    KdTree() {}
    KdTree(std::vector<mcTriangle> tris)
    {
        if (tris.empty()) return;

        std::sort(tris.begin(), tris.end(),
                [&](const mcTriangle& a, const mcTriangle& b) {
                    return getCoord(baryCenter(a), axis) < getCoord(baryCenter(b), axis);
                });
        float median = getCoord(baryCenter(tris[tris.size() / 2]), axis);

        std::vector leftTris;
        std::vector rightTris;

        for (auto tri: tris)
        {
            bbMin = fminf(bbMin, triMin(tri));
            bbMax = fmaxf(bbMax, triMax(tri));
            if (notRightOf<axis>(tri, median)) leftTris.push_back(tri);
            if (notLeftOf<axis>(tri, median)) rightTris.push_back(tri);
        }

        left = KdTree<LEVELS_LEFT-1>(leftTris);
        right = KdTree<LEVELS_LEFT-1>(rightTris);
    }

    float3 bbMin, bbMax;
    KdTree<LEVELS_LEFT-1> left, right;

    static const int axis = LEVELS_LEFT % 3;
};

struct KdTree<0>
{
    KdTree() {}
    KdTree(std::vector<mcTriangle> tris)
    {
        triangleCount = tris.size();
        cudaMalloc(&triangles, sizeof(mcTriangle) * triangleCount);
        cudaMemcpy(triangles, tris.data(), triangleCount * sizeof(mcTriangle), cudaMemcpyHostToDevice);
    }

    mcTriangle* triangles;
    size_t triangleCount;
}
