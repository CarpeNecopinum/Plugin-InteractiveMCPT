#pragma once

#include <QMutex>
#include <ACG/Math/VectorT.hh>

struct RenderTarget
{
    RenderTarget() : accumulatedColor(0), sampleCount(0), queuedSamples(0) {}
    void reset(size_t width, size_t height);

    ACG::Vec3d* accumulatedColor;
    uint32_t* sampleCount;
    uint8_t* queuedSamples;
    QMutex mutex;
};
