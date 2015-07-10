#pragma once

#include <ACG/Math/VectorT.hh>

inline ACG::Vec3d vecPow(const ACG::Vec3d& in, double exponent)
{
    return ACG::Vec3d(
                std::pow(in[0], exponent),
                std::pow(in[1], exponent),
                std::pow(in[2], exponent)
            );
}

inline ACG::Vec4f vecPow(const ACG::Vec4f& in, double exponent)
{
    return ACG::Vec4f(
                std::pow(in[0], exponent),
                std::pow(in[1], exponent),
                std::pow(in[2], exponent),
                std::pow(in[3], exponent)
            );
}
