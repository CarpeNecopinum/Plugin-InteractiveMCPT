#include "BRDF.hh"

namespace BRDF {

    ACG::Vec3d reflect(const ACG::Vec3d &incoming, const ACG::Vec3d &normal) {
        ACG::Vec3d dir = incoming.normalized();
        ACG::Vec3d reflected = dir - normal * 2.0 * (normal.normalized() | dir);

        return reflected.normalized();
    }

    Color phongSpecular(const ACG::Vec3d &incoming, const ACG::Vec3d &outgoing, const ACG::Vec3d &normal, const Material &mat)
    {
        ACG::Vec3d reflected = reflect(incoming, normal);
        double exponent = mat.shininess();
        double cosPhi = (outgoing | reflected);
        double cosTheta = (outgoing | normal);

        return mat.specularColor() * std::pow(cosPhi, exponent) / cosTheta;
    }

    Color diffuse(const Material &material)
    {
        return material.diffuseColor();
    }

    Color phongBRDF(const Material &objectMaterial, const ACG::Vec3d &incoming, const ACG::Vec3d &outgoing, const ACG::Vec3d &normal) {
        return phongSpecular(incoming, outgoing, normal, objectMaterial) + diffuse(objectMaterial);
    }
}
