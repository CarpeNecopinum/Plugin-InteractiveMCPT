#pragma once
#include <ACG/Math/VectorT.hh>
#include <stdlib.h>
#include <vector>

namespace Sampling {
    ACG::Vec3d clampToAxis(const ACG::Vec3d& n);

    double densityCosPowerTheta(ACG::Vec3d z, double exponent, ACG::Vec3d direction);
    double densityCosTheta(ACG::Vec3d z, ACG::Vec3d direction);

    std::vector<ACG::Vec3d> randomDirectionsCosTheta(int number, ACG::Vec3d n);
    std::vector<ACG::Vec3d> randomDirectionCosPowerTheta(int number, ACG::Vec3d n, double exponent);
    std::vector<ACG::Vec3d> randomDirectionsCosThetaOld(unsigned int number, ACG::Vec3d n);
    void testWeight();

    void generateTangentSystem(ACG::Vec3d &n, ACG::Vec3d &x, ACG::Vec3d &y);

    inline double random() { return ((double)rand()) / ((double)RAND_MAX);}
    inline double randomSymmetric() { return random() * 2.0 - 1.0; }

    double brightness(ACG::Vec4f color);
}
