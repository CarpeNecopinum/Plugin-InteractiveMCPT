#pragma once

class ImageViewer;
class InteractiveMCPTPlugin;

class Brush{
public:
	inline const int getSize(){ return _size; }
	inline void setSize(int size){ _size = size; }

	inline const int getDepth(){ return _depth; }
	inline void setDepth(int depth){ _depth = depth; }
private:
	int _size = 1;
	int _depth = 1;
};

class InteractiveDrawing{
	
	enum TOOLS{
		NONE,
		BRUSH
	};

public:

	inline Brush& getBrush(){ return _brush; }
	
	void toggleBrush();

	inline void deselectTool() { _aktivTools = NONE; }

	void testBrush(InteractiveMCPTPlugin* plugin, int posX, int posY);

	void update(InteractiveMCPTPlugin* plugin, ImageViewer* imageViewer);

private:
	TOOLS _aktivTools = NONE;
	Brush _brush;

};