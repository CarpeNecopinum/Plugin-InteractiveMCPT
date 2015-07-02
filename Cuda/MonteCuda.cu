#include "../MonteCuda.hh"
#include "../InfoStructs.hh"

#include <cuda.h>
#include "Cuda/cutil_math.h"
#include <vector_functions.h>
#include <iostream>


/** CUDA Part **/

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

__device__ mcIntersection mcIntersectTriangle(mcRay ray, mcTriangle& triangle)
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

__device__ float3 mcTrace(mcRay ray, int depth, mcMaterial* mats, mcTriangle* geometry, size_t triCount)
{
    if (depth > 3) return make_float3(0.0f, 0.0f, 0.0f);


    mcIntersection hit = mcIntersectScene(ray, mats, geometry, triCount);
    //Ray mirrored = reflect(_ray, hit.position, hit.normal);

    if((hit.depth == FLT_MAX) || (dot(hit.normal, -ray.direction) < 0.0))
        return make_float3(0.0f, 0.0f, 0.0f);

    return hit.material.diffuseColor;
}

__global__ void tracePixels(Point* pixels, float3* output, mcMaterial* mats, mcTriangle* geometry, mcCameraInfo* cam, size_t triCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Point& coord = pixels[idx];

    float x = float(coord.x);
    float y = float(coord.y);

    /* Ray Setup */
    float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
    mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

    /* Actual Path Tracing */

    output[idx] = mcTrace(ray, 0, mats, geometry, triCount);
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
    cudaFree(devMaterials);
    cudaMalloc(&devMaterials, sizeof(mcMaterial) * materialCount);
    cudaMemcpy(devMaterials, materials, sizeof(mcMaterial) * materialCount, cudaMemcpyHostToDevice);

    cudaFree(devTriangles);
    cudaMalloc(&devTriangles, sizeof(mcTriangle) * triCount);
    cudaMemcpy(devTriangles, tris, sizeof(mcTriangle) * triCount, cudaMemcpyHostToDevice);

    devTriangleCount = triCount;
}

void uploadCameraInfo(const CameraInfo& cam)
{
    mcCameraInfo dev;
    dev.eye_point = toCudaVec(cam.eye_point);
    dev.image_plane_start = toCudaVec(cam.image_plane_start);
    dev.x_dir = toCudaVec(cam.x_dir);
    dev.y_dir = toCudaVec(cam.y_dir);

    cudaFree(devCamera);
    cudaMalloc(&devCamera, sizeof(mcCameraInfo));
    cudaMemcpy(devCamera, &dev, sizeof(mcCameraInfo), cudaMemcpyHostToDevice);
}


void cudaTracePixels(std::vector<Point> &pixels, RenderSettings settings, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth)
{
    Point* devPixels;
    cudaMalloc(&devPixels, sizeof(Point) * pixels.size());
    cudaMemcpy(devPixels, pixels.data(), sizeof(Point) * pixels.size(), cudaMemcpyHostToDevice);

    float3* devResults;
    cudaMalloc(&devResults, sizeof(float3) * pixels.size());

    tracePixels<<<pixels.size() / 64 + 1, 64>>>(devPixels, devResults, devMaterials, devTriangles, devCamera, devTriangleCount);

    float3* hostResults = new float3[pixels.size()];
    cudaMemcpy(hostResults, devResults, sizeof(float3) * pixels.size(), cudaMemcpyDeviceToHost);

    cudaFree(devPixels);
    cudaFree(devResults);

    for (size_t i = 0; i < pixels.size(); ++i)
    {
        Point& pixel = pixels[i];
        size_t index = pixel.x + pixel.y * imageWidth;
        colorMap[index] += toACG3(hostResults[i]);
        sampleCounter[index]++;
    }

    delete[] hostResults;
}

