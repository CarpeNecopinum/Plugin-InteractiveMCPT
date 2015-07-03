#include "InteractiveDrawing.hh"

#include "InteractiveMCPT_.hh"
#include "ImageViewer.hh"

void InteractiveDrawing::update(InteractiveMCPTPlugin* plugin, ImageViewer* imageViewer){
	
}

void InteractiveDrawing::switchBrush(int type){
    _activeBrush = (BRUSHES) type;
}

void InteractiveDrawing::traceBrush(InteractiveMCPTPlugin* plugin, int posX, int posY){
    if(_activeBrush == NONE)
        return;
    InteractiveMCPTPlugin::RenderJob renderJob;
    renderJob.settings = plugin->getSettings();
    int brushSize = _brush.getSize();

    const int imageWidth = plugin->getImageViewer()->getImage()->width();
    const int imageHeight = plugin->getImageViewer()->getImage()->height();

    int currX = 0;
    int currY = 0;

    for (int offY = -brushSize; offY < brushSize; ++offY){
        for (int offX = -brushSize; offX < brushSize; ++offX){

            if((_activeBrush == CIRCLE_BRUSH || _activeBrush == GAUSSED_CIRCLE_BRUSH) &&
                    (offY * offY + offX * offX) > brushSize * brushSize)
                continue;

            currX = posX + offX;
            currY = posY + offY;

            if (currX < imageWidth && currY < imageHeight && currX >= 0 && currY >= 0){
                Point pixel = { currX, currY};
                renderJob.pixels.push_back(pixel);
            }

            if (renderJob.pixels.size() >= CUDA_BLOCK_SIZE){
                plugin->queueJob(renderJob);
                renderJob.pixels.clear();
            }
        }
    }

    if (renderJob.pixels.size() > 0)
        plugin->queueJob(renderJob);

    plugin->getUpdateTimer().start();
}
