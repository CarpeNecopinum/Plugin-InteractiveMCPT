#include "InteractiveDrawing.hh"

#include "InteractiveMCPT_.hh"
#include "ImageViewer.hh"

void InteractiveDrawing::update(InteractiveMCPTPlugin* plugin, ImageViewer* imageViewer){
	
}

void InteractiveDrawing::toggleBrush(){
	if (_aktivTools == BRUSH)
		_aktivTools = NONE;
	else
		_aktivTools = BRUSH;
}

void InteractiveDrawing::testBrush(InteractiveMCPTPlugin* plugin, int posX, int posY){

	InteractiveMCPTPlugin::RenderJob renderJob;
    renderJob.settings = plugin->getSettings();
	int brushSize = _brush.getSize();

	const int imageWidth = plugin->getImageViewer()->getImage()->width();
	const int imageHeight = plugin->getImageViewer()->getImage()->height();

	int currX = 0;
	int currY = 0;

	for (int x = -brushSize; x < brushSize; ++x){
		for (int y = -brushSize; y < brushSize; ++y){

			currX = posX + x;
			currY = posY + y;

			if (currX < imageWidth && currY < imageHeight && currX >= 0 && currY >= 0){
				Point pixel = { currX, currY};
				renderJob.pixels.push_back(pixel);
			}

            if (renderJob.pixels.size() >= 128){
				plugin->queueJob(renderJob);
				renderJob.pixels.clear();
			}
		}
	}

	if (renderJob.pixels.size() > 0)
		plugin->queueJob(renderJob);

	plugin->getUpdateTimer().start();
}
