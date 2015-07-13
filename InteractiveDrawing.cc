#include "InteractiveDrawing.hh"

#include "InteractiveMCPT_.hh"
#include "ImageViewer.hh"
#include <QMouseEvent>

#define CUDA_OPT_JOBSIZE 16 * 1024

void InteractiveDrawing::update(InteractiveMCPTPlugin* plugin, ImageViewer* imageViewer){
	
}

void InteractiveDrawing::startBrushStroke(){
    _brushStroke = true;
    _brushStrokePixels.clear();
}

void InteractiveDrawing::endBrushStroke(){

    _brushStroke = false;

    InteractiveMCPTPlugin::RenderJob renderJob;

    const int imageWidth = _plugin->getImageViewer()->getImage()->width();
    int index = 0;

    for (const QueuedPixel &qp : _brushStrokePixels){
        index = qp.y * imageWidth + qp.x;
        _plugin->getRenderTarget().paintCount[index]--;
        if (_plugin->getRenderTarget().paintCount[index] == 0){

            if (renderJob.pixels.size() >= CUDA_OPT_JOBSIZE){
                _plugin->queueJob(renderJob);
                renderJob.pixels.clear();
            }

            renderJob.pixels.push_back(qp);
        }
    }

    if (renderJob.pixels.size() > 0){
        _plugin->queueJob(renderJob);
        renderJob.pixels.clear();
    }

    QMetaObject::invokeMethod(_plugin, "updateImageWidget", Qt::QueuedConnection);
}


void InteractiveDrawing::updateBrushStroke(InteractiveMCPTPlugin *plugin, QMouseEvent* ev){

    traceBrush(plugin, ev->pos().x(), ev->pos().y());
}

void InteractiveDrawing::switchBrush(int type){
    _activeBrush = (BRUSHES) type;
}

void InteractiveDrawing::updateSigma(){
    _doubleGaussSigmaSquared = std::pow(_brush.getSize() * _sigma, 2) / 2.0;
}

void InteractiveDrawing::setSigma(double sigma) {
    _sigma = sigma;
}


void InteractiveDrawing::traceBrush(InteractiveMCPTPlugin* plugin, int posX, int posY){
    if(_activeBrush == NONE)
        return;

    int brushSize = _brush.getSize();

    const int imageWidth = plugin->getImageViewer()->getImage()->width();
    const int imageHeight = plugin->getImageViewer()->getImage()->height();

    int currX = 0;
    int currY = 0;

    double gaussScaling = gaussDistribution(0, 0);
    for (int offY = -brushSize; offY < brushSize; ++offY){
        for (int offX = -brushSize; offX < brushSize; ++offX){

            if((_activeBrush == CIRCLE_BRUSH || _activeBrush == GAUSSED_CIRCLE_BRUSH) &&
                    (offY * offY + offX * offX) > brushSize * brushSize)
                continue;

            currX = posX + offX;
            currY = posY + offY;

            if (currX < imageWidth && currY < imageHeight && currX >= 0 && currY >= 0){
                int samples = plugin->getSettings().samplesPerPixel;
                if(_activeBrush == GAUSSED_CIRCLE_BRUSH)
                    samples = (int) (gaussDistribution(offX, offY) / gaussScaling * samples);
                QueuedPixel pixel = { currX, currY, samples};
                _brushStrokePixels.push_back(pixel);
                plugin->getRenderTarget().paintCount[currX + imageWidth * currY]++;
            }
        }
    }


    QMetaObject::invokeMethod(plugin, "updateImageWidget", Qt::QueuedConnection, Q_ARG(int, posX - brushSize), Q_ARG(int, posY - brushSize), Q_ARG(int, posX + brushSize), Q_ARG(int, posY + brushSize));
}

double InteractiveDrawing::gaussDistribution(int x, int y) {
    return std::exp(-x * x / _doubleGaussSigmaSquared) *
            std::exp(-y * y / _doubleGaussSigmaSquared) /
            (M_PI * _doubleGaussSigmaSquared);
}
