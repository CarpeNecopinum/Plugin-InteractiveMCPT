#pragma once

#include <ACG/Math/VectorT.hh>

struct RenderSettings
{
    int samplesPerPixel;
};

struct CameraInfo
{
    ACG::Vec3d x_dir, y_dir;
    ACG::Vec3d image_plane_start;
    ACG::Vec3d eye_point;
};

struct Ray
{
  ACG::Vec3d origin;
  ACG::Vec3d direction;
};

struct QueuedPixel
{
    int x, y, samples;
};

