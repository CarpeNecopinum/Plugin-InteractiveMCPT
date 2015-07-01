#include "InteractiveDrawing.hh"

#include "InteractiveMCPT_.hh"
#include "ImageViewer.hh"

void InteractiveDrawing::update(InteractiveMCPTPlugin* plugin, ImageViewer* imageViewer){
	
}

void InteractiveDrawing::testJob(InteractiveMCPTPlugin* plugin, int posX, int posY){

	InteractiveMCPTPlugin::RenderJob renderJob;
	renderJob.settings.samplesPerPixel = 1;
	int brushSize = _brush.getSize();

	for (int x = 0; x < brushSize; ++x){
		for (int y = 0; y < brushSize; ++y){

			Point pixel = { posX + x, posY + y };

			if (renderJob.pixels.size() >= 64){
				plugin->queueJob(renderJob);
				renderJob.pixels.clear();
			}

			renderJob.pixels.push_back(pixel);
		}
	}

	if (renderJob.pixels.size() > 0)
		plugin->queueJob(renderJob);

	plugin->getUpdateTimer().start();
}