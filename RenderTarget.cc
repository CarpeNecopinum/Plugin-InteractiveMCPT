#include "RenderTarget.hh"

void RenderTarget::reset(size_t width, size_t height)
{
    delete[] accumulatedColor;
    ACG::Vec3d zeroVec(0.,0.,0.);
    accumulatedColor = new ACG::Vec3d[width * height];
    for (int i = 0; i < width * height; ++i)
        accumulatedColor[i] = zeroVec;

    delete[] sampleCount;
    sampleCount = new uint32_t[width * height];
    memset(sampleCount, 0, width * height * sizeof(uint32_t));

    delete[] paintCount;
    paintCount = new uint32_t[width * height];
    memset(paintCount, 0, width * height * sizeof(uint32_t));

    delete[] queuedSamples;
    queuedSamples = new uint8_t[width * height];
    memset(queuedSamples, 0, width * height * sizeof(uint8_t));
}
