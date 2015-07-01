#pragma once

class QGridLayout;
class QWidget;

class Brush{
public:
	inline const float getSize(){ return _size; }
	inline void setSize(int size){ _size = size; }
	inline void incSize(){ _size++; }
	inline void decSize(){ _size--; }
private:
	int _size = 0;
};

class InteractiveDrawing{
	
	enum TOOLS{
		NONE,
		BRUSH
	};

public:

	inline Brush& getBrush(){ return _brush; }
	inline void selectBrush(){ _aktivTools = BRUSH; }
	inline void deselectTool() { _aktivTools = NONE; }

private:
	TOOLS _aktivTools = NONE;
	Brush _brush;

};