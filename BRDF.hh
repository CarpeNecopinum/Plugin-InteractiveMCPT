#pragma once

#include <ACG/Scenegraph/MaterialNode.hh>

namespace BRDF
{
    typedef ACG::SceneGraph::Material Material;
    typedef ACG::Vec4f Color;

    ACG::Vec3d reflect(const ACG::Vec3d& incoming, const ACG::Vec3d& normal);

    Color phongSpecular(const ACG::Vec3d& incoming, const ACG::Vec3d& outgoing, const ACG::Vec3d& normal, const Material& mat);
    Color diffuse(const Material& material);

    Color phongBRDF(const Material &objectMaterial, const ACG::Vec3d &incoming, const ACG::Vec3d &outgoing, const ACG::Vec3d &normal);
}
