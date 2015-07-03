#pragma once

#include <cuda.h>
#include "cutil_math.h"
#include "../MonteCuda.hh"
#include "Geometry.hh"

//Color phongBRDF(const Material &objectMaterial, const ACG::Vec3d &incoming, const ACG::Vec3d &outgoing, const ACG::Vec3d &normal) {
__device__ float3 mcPhong(const mcMaterial& material, const float3& incoming, const float3& outgoing, const float3& normal)
{
    /*
        ACG::Vec3d reflected = reflect(incoming, normal);
        double exponent = mat.shininess();
        double cosPhi = (outgoing | reflected);
        double cosTheta = (outgoing | normal);

        return mat.specularColor() * std::pow(cosPhi, exponent) / cosTheta;
    */

    float3 reflected = mcMirror(incoming, normal);
    float exponent = material.specularExponent;
    float cosPhi = dot(outgoing, reflected);
    float cosTheta = dot(outgoing, normal);

    return pow(exponent, 0.33f) * (material.specularColor * pow(cosPhi, exponent) / cosTheta) + material.diffuseColor;
}
