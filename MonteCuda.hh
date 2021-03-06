#pragma once
#include <stdint.h>
#include <OpenFlipper/BasePlugin/PluginFunctions.hh>
#include <cuda.h>
#include <vector_functions.h>
#include <vector_types.h>
#include "InfoStructs.hh"

struct mcRectangleJob
{
	size_t left, top, width, height;
	size_t numSamples;
};

#ifndef CUDA_RECTANGLE_SIZE
#define CUDA_RECTANGLE_SIZE 32
#endif

struct mcMaterial
{
    float3 diffuseColor;
    float3 specularColor;
    float specularExponent;
    float3 emissiveColor;
};


struct mcTriangle
{
    float3 corners[3];
    uint32_t matIndex;
};

struct mcCameraInfo
{
    float3 x_dir;
    float3 y_dir;
    float3 image_plane_start;
    float3 eye_point;
};

struct mcRay
{
    float3 origin;
    float3 direction;
};

void cudaTest(void);


inline float3 toCudaVec(ACG::Vec3d acg) { return make_float3(acg[0], acg[1], acg[2]); }
inline float3 toCudaVec(ACG::Vec4f acg) { return make_float3(acg[0], acg[1], acg[2]); }
inline ACG::Vec3d toACG3(float3 cuda) {
    if (!(!isnan(cuda.x) && !isnan(cuda.y) && !isnan(cuda.z)))
    {
        std::cout << "WADDAFACK" << std::endl;
    }
    return ACG::Vec3d(cuda.x, cuda.y, cuda.z);
}

size_t cudaBlockSize();


inline mcMaterial make_material(ACG::Vec4f diffuse, ACG::Vec4f specular, double exponent, ACG::Vec4f emissive)
{
    mcMaterial result;
    result.diffuseColor = toCudaVec(diffuse);
    result.specularColor = toCudaVec(specular);
    result.emissiveColor = toCudaVec(emissive);
    result.specularExponent = exponent;
    return result;
}

struct RenderTarget;

void uploadGeometry(PluginFunctions::ObjectIterator start, PluginFunctions::ObjectIterator end);
void uploadKdTree(mcMaterial *materials, size_t materialCount, const std::vector<mcTriangle>& triangles);
void uploadBuffers(mcMaterial* materials, size_t materialCount, mcTriangle* tris, size_t triCount);
void uploadCameraInfo(const CameraInfo& cam);
void cudaTracePixels(std::vector<QueuedPixel> &pixels, RenderTarget &target, size_t imageWidth, float dsRatio);
void cudaRectangleTracePixels(mcRectangleJob& job, RenderTarget &target, size_t imageWidth, float dsRatio);
