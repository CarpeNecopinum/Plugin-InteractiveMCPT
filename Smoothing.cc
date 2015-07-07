#include "Smoothing.hh"
#include <QtConcurrent>

#include "InteractiveMCPT_.hh"
#include "ImageViewer.hh"

void Smoother::init(InteractiveMCPTPlugin *plugin){
    const int imageWidth = plugin->getImageViewer()->getImage()->width();
    const int imageHeight = plugin->getImageViewer()->getImage()->height();
    mNormals = new Vec3d[imageWidth * imageHeight];
    mDepths = new double[imageWidth * imageHeight];
    for(int i = 0; i < 7; ++i){
        mRunningFutures.push_back(QtConcurrent::run(this, &Smoother::getNormalDepth, plugin, i * imageHeight / 8, (i + 1) * imageHeight / 8));
    }
    mRunningFutures.push_back(QtConcurrent::run(this, &Smoother::getNormalDepth, plugin, 7 * imageHeight / 8, imageHeight));
}

void Smoother::getNormalDepth(InteractiveMCPTPlugin *plugin, int startY, int endY){
    const int imageWidth  = plugin->getImageViewer()->getImage()->width();

    for (int y = startY; y < endY; ++y)
    {
        for (int x = 0; x < imageWidth; ++x)
        {
            Vec3d current_point = plugin->getCam().image_plane_start + plugin->getCam().x_dir * (double) x - plugin->getCam().y_dir * (double) y;
            Ray ray = {plugin->getCam().eye_point, (current_point - plugin->getCam().eye_point).normalize()};
            InteractiveMCPTPlugin::Intersection intersection = plugin->intersectScene(ray);

            size_t index = x + y * imageWidth;
            mDepths[index] = intersection.depth;
            if(mDepths[index] == FLT_MAX){
                mNormals[index] = Vec3d(0.0, 0.0, 0.0);
            }else{
                mNormals[index] = intersection.normal.normalize();
            }
        }
    }
}

void Smoother::smooth(InteractiveMCPTPlugin *plugin, Vec3d* accumulatedColors, uint32_t* samples){
    for (std::vector<QFuture<void> >::iterator it = mRunningFutures.begin();
         it != mRunningFutures.end();){
        if(it->isFinished())
            mRunningFutures.erase(it);
        else
            it++;
    }
    if(!mRunningFutures.empty()){
        std::cout << "Still running something, please try again later." << std::endl;
        return;
    }


    const int imageHeight = plugin->getImageViewer()->getImage()->height();
    const int imageWidth = plugin->getImageViewer()->getImage()->width();
    if(mMaxDepth == 0.0){
        for (int y = 0; y < imageHeight; ++y)
        {
            for (int x = 0; x < imageWidth; ++x)
            {
                size_t index = x + y * imageWidth;
                if(mDepths[index] < FLT_MAX && mDepths[index] > mMaxDepth)
                    mMaxDepth = mDepths[index];
            }
        }
    }

    std::shared_ptr<std::vector<Vec3d>> inputColors = std::make_shared<std::vector<Vec3d>>(imageHeight * imageWidth);
    for(int i = 0; i < imageHeight * imageWidth; ++i){
        (*inputColors)[i] = accumulatedColors[i];
    }

    for(int i = 0; i < 7; ++i){
        SmoothConcurrentArgument args{plugin, inputColors, accumulatedColors, samples, i * imageHeight / 8, (i + 1) * imageHeight / 8};
        mRunningFutures.push_back(QtConcurrent::run(this, &Smoother::smoothConcurrent, args));
    }
    SmoothConcurrentArgument args{plugin, inputColors, accumulatedColors, samples, 7 * imageHeight / 8, imageHeight};
    mRunningFutures.push_back(QtConcurrent::run(this, &Smoother::smoothConcurrent, args));
}

void Smoother::smoothConcurrent(SmoothConcurrentArgument args){
    InteractiveMCPTPlugin* plugin;
    std::shared_ptr<std::vector<Vec3d>> inputColors;
    Vec3d* outputColors;
    uint32_t* samples;
    int startY;
    int endY;

    std::tie(plugin, inputColors, outputColors, samples, startY, endY) = args;

    const int imageWidth = plugin->getImageViewer()->getImage()->width();
    const int imageHeight = plugin->getImageViewer()->getImage()->height();
    double * kernel = getGaussianKernel();
    for (int y = startY; y < endY; ++y)
    {
        for (int x = 0; x < imageWidth; ++x)
        {
            size_t index = x + y * imageWidth;
            Vec3d color(0.0, 0.0, 0.0);
            double kernelSum = 0.0;
            double depth = mDepths[index];
            Vec3d normal = mNormals[index];
            if(depth == FLT_MAX)
                continue;
            for (int offY = - (int) (3 * mSigma); offY <= (int) (3 * mSigma); ++offY)
            {
                for (int offX = - (int) (3 * mSigma); offX <= (int) (3 * mSigma); ++offX)
                {
                    int currX = x + offX;
                    int currY = y + offY;
                    size_t locIndex =  currX + currY * imageWidth;

                    if (currX >= imageWidth || currY >= imageHeight || currX < 0 || currY < 0)
                        continue;

                    if(mDepths[locIndex] == FLT_MAX)
                        continue;

                    if(samples[locIndex] == 0)
                        continue;

                    if(std::acos(normal | mNormals[locIndex]) > mMaxAngleDeviation)
                        continue;
                    if(std::abs(depth - mDepths[locIndex]) > mMaxDepthDeviation * mMaxDepth)
                        continue;

                    double kernelValue = kernel[offY + (int) (3 * mSigma)] * kernel[offX + (int) (3 * mSigma)];
                    kernelSum += kernelValue;
                    color += kernelValue * (*inputColors)[locIndex] / samples[locIndex];
                }
            }
            if(kernelSum > 0){
                color /= kernelSum;
                color *= samples[index];
                outputColors[index] = color;
            }
        }
    }
    delete kernel;
    plugin->updateImageWidget();
}

void Smoother::setMaxAngleDeviation(double maxAngleDev){
    mMaxAngleDeviation = maxAngleDev;
}

void Smoother::setMaxDepthDeviation(double maxDepthDev){
    mMaxAngleDeviation = maxDepthDev;
}

void Smoother::setSigma(double sigma){
    mSigma = sigma;
}


double* Smoother::getGaussianKernel(){
    double * kernel = new double[(int)(6 * mSigma) + 1];
    double doubleGaussSigmaSquared = 2 * mSigma * mSigma;
    for (int x = - (int) (3 * mSigma); x <= (int) (3 * mSigma); ++x)
    {
        size_t index = x + (int) (3 * mSigma);
        kernel[index] = std::exp(-x * x / doubleGaussSigmaSquared) /
                (std::sqrt(2 * M_PI) * mSigma);
    }
    return kernel;
}
