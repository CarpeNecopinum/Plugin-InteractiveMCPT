#pragma once

#include <QObject>

#include <InfoStructs.hh>
#include <map>

class QMouseEvent;
class ImageViewer;
class InteractiveMCPTPlugin;


class Brush{
public:
    inline int getSize() const { return _size; }
	inline void setSize(int size){ _size = size; }

private:
	int _size = 1;
};

class InteractiveDrawing : public QObject{
	
	Q_OBJECT

    enum BRUSHES{
        NONE = 0,
        SQUARE_BRUSH = 1,
        CIRCLE_BRUSH = 2,
        GAUSSED_CIRCLE_BRUSH = 3
	};

public:


    InteractiveDrawing(InteractiveMCPTPlugin *plugin) : _plugin(plugin){}

	inline Brush& getBrush(){ return _brush; }
    inline BRUSHES getActiveBrush() {return _activeBrush;}
	
    void switchBrush(int type);

    void updateSigma();

    void setSigma(double sigma);

    void traceBrush(InteractiveMCPTPlugin* plugin, int posX, int posY);

	void update(InteractiveMCPTPlugin* plugin, ImageViewer* imageViewer);

    double gaussDistribution(int x, int y);

    void startBrushStroke();
    void endBrushStroke();

    void updateBrushStroke(InteractiveMCPTPlugin* plugin, QMouseEvent* ev);

    inline bool isPerformingStroke(){return _brushStroke;}
    inline std::vector<QueuedPixel> &getBrushStrokePixels(){ return _brushStrokePixels;}

private:

    BRUSHES _activeBrush = NONE;
	Brush _brush;
    double _sigma = 0.75, _doubleGaussSigmaSquared = _brush.getSize() * _sigma * _brush.getSize() * _sigma / 2.0;

    bool _brushStroke = false;

    std::vector<QueuedPixel> _brushStrokePixels = {};

    InteractiveMCPTPlugin *_plugin = 0;

    int _brushMinX = 0;
    int _brushMinY = 0;
    int _brushMaxX = 0;
    int _brushMaxY = 0;
};
