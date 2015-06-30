#include <stdint.h>
#include <OpenFlipper/BasePlugin/PluginFunctions.hh>
#include <cuda.h>
#include "InfoStructs.hh"

void cudaTest(void);

struct mcMaterial
{
    mcMaterial(ACG::Vec4f diffuse, ACG::Vec4f specular, double exponent, ACG::Vec4f emissive)
    {
        for (int i = 0; i < 3; i++)
        {
            diffuseColor[i] = diffuse[i];
            specularColor[i] = specular[i];
            emissiveColor[i] = emissive[i];
        }
        specularExponent = exponent;
    }

    float diffuseColor[3];
    float specularColor[3];
    float specularExponent;
    float emissiveColor[3];
};

struct mcVertex
{
    mcVertex() {}
    mcVertex(ACG::Vec3d point)
    {
        x = (float)point[0];
        y = (float)point[1];
        z = (float)point[2];
    }

    float x,y,z;
};

struct mcTriangle
{
    mcVertex corners[3];
    uint32_t matIndex;
};

struct mcCameraInfo
{
    float x_dir[3];
    float y_dir[3];
    float image_plane_start[3];
    float eye_point[3];
};


void uploadGeometry(PluginFunctions::ObjectIterator start, PluginFunctions::ObjectIterator end);
void uploadBuffers(mcMaterial* materials, size_t materialCount, mcTriangle* tris, size_t triCount);
void uploadCameraInfo(const CameraInfo& cam);
void cudaTracePixels(std::vector<Point> &pixels, RenderSettings settings, ACG::Vec3d* colorMap, uint32_t* sampleCounter, size_t imageWidth);
