#include "../MonteCuda.hh"
#include "../InfoStructs.hh"

#include <cuda.h>
#include "Cuda/cutil_math.h"
#include <curand_kernel.h>
#include <vector_functions.h>
#include <iostream>

#define CUDA_CHECK cudaCheck(__LINE__);


void cudaCheck(int line)
{
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error %i: %s\n", line, cudaGetErrorString(error));
        exit(-1);
    }
}

/** CUDA Part **/


__device__ float mcBrightness(const float3& color) {
    // Convert to Y (Yuv Color space)
    return (0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
}

__device__ float3 mcMirror(const float3& direction, const float3& normal)
{
    return direction - normal * 2.f * dot(normal, direction);
}

__global__ void fillWithOne(int* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 1;
}

struct mcIntersection
{
    float3 position;
    float3 normal;
    float depth;
    mcMaterial material;
};

__device__ float3 clampToAxis(const float3 &n) {
    if(abs(n.x) > std::fabs(n.y) && std::fabs(n.x) > std::fabs(n.z)) {
        return make_float3(0,0,1);
    } else if(std::fabs(n.y) > std::fabs(n.x) && std::fabs(n.y) > std::fabs(n.z)) {
        return make_float3(1,0,0);
    }
    return make_float3(0,1,0);
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

__device__ float3 mcPhong(const float3& incoming, const float3& outgoing, const float3& normal, mcMaterial& mat)
{
    float3 reflected = mcMirror(incoming, normal);
    float exponent = mat.specularExponent;
    float cosPhi = dot(outgoing, reflected);
    float cosTheta = dot(outgoing, normal);

    return mat.specularColor * pow(cosPhi, exponent) / cosTheta + mat.diffuseColor;
}

__device__ float3 mcRandomDirCosPowerTheta(float3 normal, float exponent, curandState* state )
{
    // Generate Tangent System
    float3 y = clampToAxis(normal);
    float3 x = normalize(cross(normal, y));
    y = normalize(cross(x, normal));

    float xi1 = curand_uniform(state);
    float xi2 = curand_uniform(state);

    float costheta = pow(xi1, 1.0f / (exponent + 1.0f));
    float theta = acos(costheta);
    float phi = 2.0f * M_PI * xi2;

    return
        x * cos(phi) * sin(theta) +
        y * sin(phi) * sin(theta) +
        normal * cos(theta);
}

__device__ float3 mcRandomDirCosTheta(float3 normal, curandState* state)
{
    return mcRandomDirCosPowerTheta(normal, 1.0f, state);
}

__device__ bool mcBaryTest(float3 point, mcTriangle& triangle)
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

__device__ mcIntersection mcIntersectTriangle(mcRay& ray, mcTriangle& triangle)
{
    mcIntersection result;
    result.depth = FLT_MAX;

    result.normal = normalize(cross(triangle.corners[1] - triangle.corners[0], triangle.corners[2] - triangle.corners[0]));
    float depth = dot(triangle.corners[0] - ray.origin, result.normal) / dot(ray.direction, result.normal);
    if (depth <= 0.001f) { return result; }

    float3 intersection = ray.origin + depth * ray.direction;

    if (mcBaryTest(intersection, triangle)) result.depth = depth;

    return result;
}

__device__ mcIntersection mcIntersectScene(mcRay& ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount)
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


__device__ float3 mcTriangleNormal(mcTriangle& tri)
{
    return normalize(cross(tri.corners[1] - tri.corners[0], tri.corners[2] - tri.corners[0]));
}

template<int i>
__device__ float3 mcTrace(mcRay& ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount, curandState* state)
{    
    mcIntersection hit = mcIntersectScene(ray, mats, geometry, triCount);

    if((hit.depth == FLT_MAX) || (dot(hit.normal, -ray.direction) < 0.f))
        return make_float3(0.0f, 0.0f, 0.0f);

    float3 mirrored = mcMirror(ray.direction, hit.normal);

    float diffuseReflectance = mcBrightness(hit.material.diffuseColor);
    float specularReflectance = mcBrightness(hit.material.specularColor);
    float totalReflectance = diffuseReflectance + specularReflectance;

    float3 sample;
    ((curand_uniform(state) * totalReflectance) <= diffuseReflectance)
        ? sample = mcRandomDirCosTheta(hit.normal, state)
        : sample = mcRandomDirCosPowerTheta(mirrored, hit.material.specularExponent, state);
    float density = (diffuseReflectance / totalReflectance) * mcDensityCosTheta(hit.normal, sample)
                  + (specularReflectance / totalReflectance) * mcDensityCosPowerTheta(mirrored, hit.material.specularExponent, sample);


    float costheta = dot(sample, hit.normal);

    mcRay reflectedRay = {hit.position, sample};
    float3 reflected = mcPhong(ray.direction, reflectedRay.direction, hit.normal, hit.material)
            * mcTrace<i-1>(reflectedRay, mats, geometry, triCount, state) * costheta / density;

    return hit.material.emissiveColor + reflected;
}

template<>
__device__ float3 mcTrace<0>(mcRay& ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount, curandState* state)
{
    return make_float3(0.f, 0.f, 0.f);
}


__device__ float cudaRandomSymmetric(curandState* state) { return curand_uniform(state) * 2.f - 1.f; }

__global__ void tracePixels(Point* pixels, float3* output, mcMaterial* mats, mcTriangle* geometry, mcCameraInfo* cam, size_t triCount, RenderSettings settings, uint32_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Point& coord = pixels[idx];

    if (coord.x == -1) return;

    curandState state;
    curand_init ( seed, idx, 0, &state );

    float x = float(coord.x) + .5f * cudaRandomSymmetric(&state);
    float y = float(coord.y) + .5f * cudaRandomSymmetric(&state);

    /* Ray Setup */
    float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
    mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

    /* Actual Path Tracing */

    output[idx] = mcTrace<3>(ray, mats, geometry, triCount, &state);
}

/** C(++) Part **/

mcTriangle*  devTriangles = 0;
size_t devTriangleCount = 0;
mcMaterial*  devMaterials = 0;
mcCameraInfo*   devCamera = 0;

void cudaTest(void)
{
    const size_t length = 64;
    int data[length];

    int* devPtr;
    cudaMalloc(&devPtr, length * sizeof(int));

    fillWithOne<<<4, 16>>>(devPtr);

    cudaMemcpy(data, devPtr, sizeof(int) * length, cudaMemcpyDeviceToHost);

    for (size_t i = 0u; i < length; i++)
        std::cout << data[i] << std::endl;
}

void uploadBuffers(mcMaterial *materials, size_t materialCount, mcTriangle *tris, size_t triCount)
{
    cudaFree(devMaterials); CUDA_CHECK
    cudaMalloc(&devMaterials, sizeof(mcMaterial) * materialCount); CUDA_CHECK
    cudaMemcpy(devMaterials, materials, sizeof(mcMaterial) * materialCount, cudaMemcpyHostToDevice); CUDA_CHECK

    cudaFree(devTriangles); CUDA_CHECK
    cudaMalloc(&devTriangles, sizeof(mcTriangle) * triCount); CUDA_CHECK
    cudaMemcpy(devTriangles, tris, sizeof(mcTriangle) * triCount, cudaMemcpyHostToDevice); CUDA_CHECK

    devTriangleCount = triCount;
}

void uploadCameraInfo(const CameraInfo& cam)
{
    mcCameraInfo dev;
    dev.eye_point = toCudaVec(cam.eye_point);
    dev.image_plane_start = toCudaVec(cam.image_plane_start);
    dev.x_dir = toCudaVec(cam.x_dir);
    dev.y_dir = toCudaVec(cam.y_dir);

    cudaFree(devCamera); CUDA_CHECK
    cudaMalloc(&devCamera, sizeof(mcCameraInfo)); CUDA_CHECK
    cudaMemcpy(devCamera, &dev, sizeof(mcCameraInfo), cudaMemcpyHostToDevice); CUDA_CHECK
}


void cudaTracePixels(std::vector<Point> &pixels, RenderSettings settings, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth)
{
    Point* devPixels;
    cudaMalloc(&devPixels, sizeof(Point) * pixels.size()); CUDA_CHECK
    cudaMemcpy(devPixels, pixels.data(), sizeof(Point) * pixels.size(), cudaMemcpyHostToDevice); CUDA_CHECK

    float3* devResults;
    cudaMalloc(&devResults, sizeof(float3) * pixels.size()); CUDA_CHECK

    assert(pixels.size() % CUDA_BLOCK_SIZE == 0);

    std::cout << "Tracing....." << pixels.size() << std::endl;

    tracePixels<<<pixels.size() / cudaBlockSize(), cudaBlockSize()>>>(devPixels, devResults, devMaterials, devTriangles, devCamera, devTriangleCount, settings, rand() << 16 | rand()); CUDA_CHECK

    float3* hostResults = new float3[pixels.size()];
    cudaMemcpy(hostResults, devResults, sizeof(float3) * pixels.size(), cudaMemcpyDeviceToHost); CUDA_CHECK

    cudaFree(devPixels); CUDA_CHECK
    cudaFree(devResults); CUDA_CHECK

    for (size_t i = 0; i < pixels.size(); ++i)
    {
        Point& pixel = pixels[i];
        size_t index = pixel.x + pixel.y * imageWidth;
        colorMap[index] += toACG3(hostResults[i]);
        sampleCounter[index]++;
    }

    delete[] hostResults;
}

