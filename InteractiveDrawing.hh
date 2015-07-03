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
	
    enum BRUSHES{
        NONE = 0,
        SQUARE_BRUSH = 1,
        CIRCLE_BRUSH = 2,
        GAUSSED_CIRCLE_BRUSH = 3
	};

public:

	inline Brush& getBrush(){ return _brush; }
	
    void switchBrush(int type);

    void traceBrush(InteractiveMCPTPlugin* plugin, int posX, int posY);

	void update(InteractiveMCPTPlugin* plugin, ImageViewer* imageViewer);

private:
    BRUSHES _activeBrush = NONE;
	Brush _brush;

};
