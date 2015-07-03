#pragma once

#include <cuda.h>
#include "cutil_math.h"
#include <curand_kernel.h>
#include "../MonteCuda.hh"
#include "Geometry.hh"

__device__ float mcDensityCosPowerTheta(const float3& z, float exponent, const float3& direction);
__device__ float mcDensityCosTheta(float3 z, float3 direction);
__device__ float3 mcRandomDirCosPowerTheta(float3 normal, float exponent, curandState* state );
__device__ float3 mcRandomDirCosTheta(float3 normal, curandState* state);
__device__ void generateTangentSystem(float3 &n, float3 &x, float3 &y);


__device__ void generateTangentSystem(float3 &n, float3 &x, float3 &y)
{
    n = normalize(n);
    y = clampToAxis(n);
    x = cross(n, y);
    y = cross(x, n);
    x = normalize(x);
    y = normalize(y);
}


__device__ float mcDensityCosPowerTheta(const float3& z, float exponent, const float3& direction)
{
    float costheta = dot(z, direction);
    float theta = acos(costheta);
    return (exponent + 1.0) * pow(costheta, exponent) * sin(theta);
}

__device__ float mcDensityCosTheta(float3 z, float3 direction)
{
    return mcDensityCosPowerTheta(z, 1.f, direction);
}

__device__ float3 mcRandomDirCosPowerTheta(float3 normal, float exponent, curandState* state )
{
    float3 x_dir, y_dir;
    generateTangentSystem(normal, x_dir, y_dir);

    float xi1 = curand_uniform(state);
    float costheta = pow(xi1, 1.0f / (exponent + 1.0f));
    float theta = acos(costheta);

    float xi2 = curand_uniform(state);
    float phi = 2.0f * M_PI * xi2;

    float3 direction = x_dir * cos(phi) * sin(theta) +
                       y_dir * sin(phi) * sin(theta) +
                       normal * cos(theta);
    return normalize(direction);
}

__device__ float3 mcRandomDirCosTheta(float3 normal, curandState* state)
{
    return mcRandomDirCosPowerTheta(normal, 1.0f, state);
}

__device__ float3 mcRandomDirUniform(float3 normal, curandState* state)
{
    float xi1 = curand_uniform(state);
    float r = sqrt(1.f - xi1 * xi1);
    float phi = 2.f * M_PI * curand_uniform(state);
    return normalize(make_float3(cos(phi) * r, sin(phi) * r, xi1));
}
