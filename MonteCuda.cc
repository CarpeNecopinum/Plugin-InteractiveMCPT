#include "MonteCuda.hh"

#include <vector>
#include <ObjectTypes/TriangleMesh/TriangleMesh.hh>
#include <ObjectTypes/TriangleMesh/TriangleMeshTypes.hh>
#include <ObjectTypes/TriangleMesh/PluginFunctionsTriangleMesh.hh>



mcTriangle makeTriangle(TriMesh::FaceHandle fh, TriMesh& mesh, uint32_t material)
{
    mcTriangle result;

    TriMesh::FaceVertexIter fv_it(mesh, fh);
    for (int i = 0; i < 3; i++)
    {
        result.corners[i] = toCudaVec(mesh.point(*fv_it));
        ++fv_it;
    }
    result.matIndex = material;

    return result;
}


void uploadGeometry(PluginFunctions::ObjectIterator start, PluginFunctions::ObjectIterator end)
{
    std::vector<mcTriangle> triangles;
    std::vector<mcMaterial> materials;

    for (; start != end; ++start)
    {
        BaseObjectData& object = **start;
        TriMesh& mesh = *( PluginFunctions::triMeshObject(&object)->mesh() );

        // Save material
        ACG::SceneGraph::Material material = object.materialNode()->material();
        mcMaterial mat = make_material(
                    material.diffuseColor(),
                    material.specularColor(),
                    material.shininess(),
                    float(material.reflectance()) * ACG::Vec4f(1.0, 1.0, 1.0, 1.0)
                );
        materials.push_back(mat);

        // Save triangles
        TriMesh::FaceIter f_it( mesh.faces_begin() );
        TriMesh::FaceIter f_end( mesh.faces_end() );
        for (; f_it != f_end; ++f_it)
        {
            triangles.push_back(makeTriangle(*f_it, mesh, materials.size() - 1));
        }
    }
    uploadBuffers(materials.data(), materials.size(), triangles.data(), triangles.size());
}
