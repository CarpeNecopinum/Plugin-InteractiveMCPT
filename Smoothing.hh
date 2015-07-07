#pragma once

#include <OpenFlipper/common/Types.hh>
#include <QFuture>
#include <memory>
#include <tuple>

class InteractiveMCPTPlugin;

typedef ACG::Vec3d Vec3d;


struct Point
{
    int x, y;
};

struct SmoothJob
{
    std::vector<Point> pixels;
};

typedef std::tuple<InteractiveMCPTPlugin*, std::shared_ptr<std::vector<Vec3d>>, Vec3d*, uint32_t*, int, int> SmoothConcurrentArgument;

class Smoother{
public:
    void init(InteractiveMCPTPlugin* plugin);

    void getNormalDepth(InteractiveMCPTPlugin* plugin, int startX, int endX);

    void smooth(InteractiveMCPTPlugin *plugin, Vec3d* accumulatedColors, uint32_t* samples);

    void smoothConcurrent(SmoothConcurrentArgument args);

    void setMaxAngleDeviation(double maxAngleDev);

    void setMaxDepthDeviation(double maxDepthDev);

    void setSigma(double sigma);

    double* getGaussianKernel();

private:
    Vec3d * mNormals;
    double * mDepths;
    double mMaxAngleDeviation = 0.052, mMaxDepthDeviation = 0.05, mMaxDepth = 0.0, mSigma = 5.00;
    std::vector<QFuture<void>> mRunningFutures;
};
