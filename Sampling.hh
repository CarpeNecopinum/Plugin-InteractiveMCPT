#pragma once
#include <ACG/Math/VectorT.hh>
#include <stdlib.h>
#include <vector>

namespace Sampling {
    ACG::Vec3d clampToAxis(const ACG::Vec3d& n);
    std::vector<ACG::Vec3d> randomDirectionsCosTheta(int number, ACG::Vec3d n);
    std::vector<ACG::Vec3d> randomDirectionCosPowerTheta(int number, ACG::Vec3d n, double exponent);

    void generateTangentSystem(ACG::Vec3d &n, ACG::Vec3d &x, ACG::Vec3d &y);

    inline double random() { return ((double)rand()) / ((double)RAND_MAX);}
    inline double randomSymmetric() { return random() * 2.0 - 1.0; }
}
