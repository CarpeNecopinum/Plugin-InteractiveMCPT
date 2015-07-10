#include "../MonteCuda.hh"
#include "../InfoStructs.hh"

#include <cuda.h>
#include "Cuda/cutil_math.h"
#include <curand_kernel.h>
#include <vector_functions.h>
#include <iostream>

#include "Geometry.hh"
#include "Sampling.hh"
#include "BRDF.hh"

#include "KdTree.hh"

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

/*
__device__ float3 mcTrace(mcRay ray, mcMaterial* mats, mcTriangle* geometry, size_t triCount, curandState* state)
{
    mcIntersection hit = mcIntersectScene(ray, mats, geometry, triCount);

    float3 emitted = hit.material.emissiveColor;

    if((hit.depth == FLT_MAX) || (dot(hit.normal, -ray.direction) < 0.f))
        return make_float3(0.0f, 0.0f, 0.0f);

    float diffuseReflectance = mcBrightness(hit.material.diffuseColor);
    float specularReflectance = mcBrightness(hit.material.specularColor);
    float totalReflectance = diffuseReflectance + specularReflectance;

    float3 mirrored = mcMirror(ray.direction, hit.normal);


    float3 sample;
    ((curand_uniform(state) * totalReflectance) <= diffuseReflectance)
        ? sample = mcRandomDirCosTheta(hit.normal, state)
        : sample = mcRandomDirCosPowerTheta(mirrored, hit.material.specularExponent, state);
    float density = (diffuseReflectance / totalReflectance) * mcDensityCosTheta(hit.normal, sample)
                  + (specularReflectance / totalReflectance) * mcDensityCosPowerTheta(mirrored, hit.material.specularExponent, sample);

    mcRay reflectedRay = {hit.position, sample};
    float costheta = dot(sample, hit.normal);

    float3 reflected = mcPhong(hit.material, ray.direction, reflectedRay.direction, hit.normal) *
             mcTrace(reflectedRay, mats, geometry, triCount, state) * costheta / density;

    return reflected + emitted;
}*/

/*
template<int RECURSIONS_LEFT>
__device__ float3 mcTrace(mcRay ray, mcMaterial* mats, KdTree<TREE_DEPTH>* tree, curandState* state)
{
    mcIntersection hit = mcIntersectKdTree(ray, mats, tree);


    float3 emitted = hit.material.emissiveColor;

    if((hit.depth == FLT_MAX) || (dot(hit.normal, -ray.direction) < 0.f))
        return make_float3(0.0f, 0.0f, 0.0f);

    float diffuseReflectance = mcBrightness(hit.material.diffuseColor);
    float specularReflectance = mcBrightness(hit.material.specularColor);
    float totalReflectance = diffuseReflectance + specularReflectance;

    float3 mirrored = mcMirror(ray.direction, hit.normal);


    float3 sample;
    ((curand_uniform(state) * totalReflectance) <= diffuseReflectance)
        ? sample = mcRandomDirCosTheta(hit.normal, state)
        : sample = mcRandomDirCosPowerTheta(mirrored, hit.material.specularExponent, state);
    float density = (diffuseReflectance / totalReflectance) * mcDensityCosTheta(hit.normal, sample)
                  + (specularReflectance / totalReflectance) * mcDensityCosPowerTheta(mirrored, hit.material.specularExponent, sample);

    mcRay reflectedRay = {hit.position, sample};
    float costheta = dot(sample, hit.normal);

    float3 reflected = mcPhong(hit.material, ray.direction, reflectedRay.direction, hit.normal) *
             mcTrace<RECURSIONS_LEFT-1>(reflectedRay, mats, tree, state) * costheta / density;

    return reflected + emitted;
}

template<>
__device__ float3 mcTrace<0>(mcRay ray, mcMaterial* mats, KdTree<TREE_DEPTH>* tree, curandState* state)
{
    return make_float3(0.f, 0.f, 0.f);
}
*/

__device__ float3 mcTrace(mcRay ray, mcMaterial* mats, mcTriangle* tris, size_t triCount, curandState* state)
{
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 weightLeft = make_float3(1.0f, 1.0f, 1.0f);

    int bounces = -3;
    while (bounces < 1 && (weightLeft.x > 0.02f || weightLeft.y > 0.01f || weightLeft.z > 0.05f))
    {
        mcIntersection hit = mcIntersectScene(ray, mats, tris, triCount);

        if((hit.depth == FLT_MAX) || (dot(hit.normal, -ray.direction) < 0.f))
            return color;

        color += weightLeft * hit.material.emissiveColor;

        if (bounces < 0)
        {
            float diffuseReflectance = mcBrightness(hit.material.diffuseColor);
            float specularReflectance = mcBrightness(hit.material.specularColor);
            float totalReflectance = diffuseReflectance + specularReflectance;

            float3 mirrored = mcMirror(ray.direction, hit.normal);

            float3 sample;
            ((curand_uniform(state) * totalReflectance) <= diffuseReflectance)
                ? sample = mcRandomDirCosTheta(hit.normal, state)
                : sample = mcRandomDirCosPowerTheta(mirrored, hit.material.specularExponent, state);
            float density = (diffuseReflectance / totalReflectance) * mcDensityCosTheta(hit.normal, sample)
                          + (specularReflectance / totalReflectance) * mcDensityCosPowerTheta(mirrored, hit.material.specularExponent, sample);

            mcRay reflectedRay = {hit.position, sample};
            float costheta = dot(sample, hit.normal);

            weightLeft = weightLeft * mcPhong(hit.material, ray.direction, reflectedRay.direction, hit.normal) * costheta / density;
            ray = reflectedRay;
        }
        bounces++;
    }

    return color;
}


__device__ float cudaRandomSymmetric(curandState* state) { return curand_uniform(state) * 2.f - 1.f; }

/*
__global__ void tracePixels(QueuedPixel* pixels, float3* output, mcMaterial* mats, KdTree<TREE_DEPTH>* tree, mcCameraInfo* cam, uint32_t seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    QueuedPixel& pixel = pixels[idx];

    if (pixel.x == -1) return;

    curandState state;
    curand_init ( seed, idx, 0, &state );

    float3 color = make_float3(0.f, 0.f, 0.f);

    const int samples = min(pixel.samples, 10);
    pixel.samples -= samples;
    for (int i = 0; i < samples; i++)
    {
        float x = float(pixel.x) + .5f * cudaRandomSymmetric(&state);
        float y = float(pixel.y) + .5f * cudaRandomSymmetric(&state);

        // Ray Setup
        float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
        mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

        // Actual Path Tracing
        color += mcTrace(ray, mats, tree, &state);
    }
    output[idx] = color;
}*/

__global__ void tracePixels(QueuedPixel* pixels, float3* output, mcMaterial* mats, mcTriangle* tris, size_t triCount, mcCameraInfo* cam, uint32_t seed, bool firstRun)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    QueuedPixel& pixel = pixels[idx];

    if (pixel.x == -1) return;

    curandState state;
    curand_init ( seed, idx, 0, &state );

    float3 color = make_float3(0.f, 0.f, 0.f);

    const int samples = min(pixel.samples, 10);
    for (int i = 0; i < samples; i++)
    {
        float x = float(pixel.x) + .5f * cudaRandomSymmetric(&state);
        float y = float(pixel.y) + .5f * cudaRandomSymmetric(&state);

        /* Ray Setup */
        float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
        mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

        /* Actual Path Tracing */
        float3 result = mcTrace(ray, mats, tris, triCount, &state);
        if (!isnan(result.x) && !isnan(result.y) && !isnan(result.z)
         && !isinf(result.x) && !isinf(result.y) && !isinf(result.z))
        {
            color += result;
            pixel.samples--;
        }
    }
    if (firstRun)
        output[idx] = color;
        else output[idx] += color;
}

__global__ void tracePixelsRect(mcRectangleJob* job, float3* output, mcMaterial* mats, mcTriangle* tris, size_t triCount, mcCameraInfo* cam, uint32_t seed, bool firstRun)
{
    size_t blockX = threadIdx.x + blockIdx.x * blockDim.x;
    size_t blockY = threadIdx.y + blockIdx.y * blockDim.y;

    if (blockX > job->width || blockY > job->height) return;
    size_t idx = blockX + job->width * blockY;

    curandState state;
    curand_init ( seed, idx, 0, &state );

    float3 color = make_float3(0.f, 0.f, 0.f);

    uint2 pixel = {blockX + job->left, blockY + job->top};

    const int samples = min(uint(job->numSamples), 10u);
    for (int i = 0; i < samples;)
    {
        float x = float(pixel.x) + .5f * cudaRandomSymmetric(&state);
        float y = float(pixel.y) + .5f * cudaRandomSymmetric(&state);

        // Ray Setup //
        float3 current_point = cam->image_plane_start + cam->x_dir * x - cam->y_dir * y;
        mcRay ray = {cam->eye_point, normalize(current_point - cam->eye_point)};

        // Actual Path Tracing //
        float3 result = mcTrace(ray, mats, tris, triCount, &state);
        if (!isnan(result.x) && !isnan(result.y) && !isnan(result.z))
        {
            color += result;
            i++;
        }
    }
    if (firstRun)
        output[idx] = color;
        else output[idx] += color;
}
/** C(++) Part **/

mcTriangle*  devTriangles = 0;
size_t devTriangleCount = 0;
mcMaterial*  devMaterials = 0;
mcCameraInfo*   devCamera = 0;

KdTree<TREE_DEPTH>* devKdTree = 0;

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

void uploadKdTree(mcMaterial *materials, size_t materialCount, const std::vector<mcTriangle>& triangles)
{/*
    cudaFree(devKdTree);
    cudaMalloc(&devKdTree, sizeof(KdTree<TREE_DEPTH>));

    KdTree<TREE_DEPTH> hostTree(triangles);
    cudaMemcpy(devKdTree, &hostTree, sizeof(KdTree<TREE_DEPTH>), cudaMemcpyHostToDevice);

    cudaFree(devMaterials); CUDA_CHECK
    cudaMalloc(&devMaterials, sizeof(mcMaterial) * materialCount); CUDA_CHECK
    cudaMemcpy(devMaterials, materials, sizeof(mcMaterial) * materialCount, cudaMemcpyHostToDevice); CUDA_CHECK
*/
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


void cudaTracePixels(std::vector<QueuedPixel> &pixels, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth)
{
    QueuedPixel* devPixels;
    cudaMalloc(&devPixels, sizeof(QueuedPixel) * pixels.size()); CUDA_CHECK
    cudaMemcpy(devPixels, pixels.data(), sizeof(QueuedPixel) * pixels.size(), cudaMemcpyHostToDevice); CUDA_CHECK

    size_t num_passes = 0;
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        num_passes = std::max(size_t(pixels[i].samples + 10 - 1) / 10, num_passes);
    }

    float3* devResults;
    cudaMalloc(&devResults, sizeof(float3) * pixels.size()); CUDA_CHECK

    assert(pixels.size() % CUDA_BLOCK_SIZE == 0);
    float3* hostResults = new float3[pixels.size()];

    bool firstRun = true;
    for (size_t pass = 0; pass < num_passes; ++pass)
    {
        tracePixels<<<pixels.size() / cudaBlockSize(), cudaBlockSize()>>>(devPixels, devResults, devMaterials, devTriangles, devTriangleCount, devCamera, rand() << 16 | rand(), firstRun); CUDA_CHECK
        firstRun = false;
    }

    cudaMemcpy(hostResults, devResults, sizeof(float3) * pixels.size(), cudaMemcpyDeviceToHost); CUDA_CHECK
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        QueuedPixel& pixel = pixels[i];
        if (pixel.samples > 0)
        {
            size_t index = pixel.x + pixel.y * imageWidth;
            colorMap[index] += toACG3(hostResults[i]);
            sampleCounter[index] += pixel.samples;
        }
    }

    cudaFree(devPixels); CUDA_CHECK
    cudaFree(devResults); CUDA_CHECK

    delete[] hostResults;
}


void cudaRectangleTracePixels(mcRectangleJob& job, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth)
{
    size_t num_passes = size_t(job.numSamples + 10 - 1) / 10;
    const int originalNumSamples = job.numSamples;

    float3* devResults;
    cudaMalloc(&devResults, sizeof(float3) * job.width * job.height); CUDA_CHECK

    mcRectangleJob* devJob;
    cudaMalloc(&devJob, sizeof(mcRectangleJob));
    cudaMemcpy(devJob, &job, sizeof(mcRectangleJob), cudaMemcpyHostToDevice);

    float3* hostResults = new float3[job.width * job.height];

    dim3 blockSize(CUDA_RECTANGLE_SIZE, CUDA_RECTANGLE_SIZE);
    dim3 gridSize((job.width + CUDA_RECTANGLE_SIZE -1) / CUDA_RECTANGLE_SIZE, (job.height + CUDA_RECTANGLE_SIZE -1) / CUDA_RECTANGLE_SIZE);

    bool firstRun = true;
    for (size_t pass = 0; pass < num_passes; ++pass)
    {
        tracePixelsRect<<<gridSize, blockSize>>>(devJob, devResults, devMaterials, devTriangles, devTriangleCount, devCamera, rand() << 16 | rand(), firstRun); CUDA_CHECK
        job.numSamples -= 10;
        cudaMemcpy(devJob, &job, sizeof(mcRectangleJob), cudaMemcpyHostToDevice);
        firstRun = false;
    }

    cudaMemcpy(hostResults, devResults, sizeof(float3) * job.width * job.height, cudaMemcpyDeviceToHost); CUDA_CHECK


    for (size_t y = 0; y < job.height; y++)
    for (size_t x = 0; x < job.width; x++)
    {
        size_t index = (x + job.left) + (y + job.top) * imageWidth;
        colorMap[index] += toACG3(hostResults[x + job.width * y]);
        sampleCounter[index] += originalNumSamples;
    }

    cudaFree(devResults); CUDA_CHECK

    delete[] hostResults;
}



