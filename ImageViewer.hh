#pragma once

#include <QLabel>
#include <InteractiveMCPT_.hh>

#include <QCursor>

class ImageViewer : public QLabel{
    Q_OBJECT

signals:
    void mousePressed(QMouseEvent* ev);
	void mouseReleased(QMouseEvent* ev);
	void mouseMoved(QMouseEvent* ev);
	void mouseLeaved(QEvent* ev);
	void mouseEntered(QEvent* ev);

public:

    ImageViewer(InteractiveMCPTPlugin* _plugin, QImage* image, QWidget* parent = 0) : QLabel(parent), image_(image), plugin(_plugin) {
        _defaultBrushPixmap = QBitmap(1,1);
        _defaultBrushPixmap.fill(Qt::black);
        updateCursorIcon();
        changeCursor(0);
    }

	inline bool isMouseDown(){ return mIsMouseDown; }
    inline bool isFocused(){ return mIsFocused; }

	QImage* getImage(){ return image_; }

    void updateCursorIcon();

    void changeCursor(int type);

    inline const QPixmap &getCircleBrushPixmap() {return _circleBrushPixmap;}
    inline const QPixmap &getSquarBrushPixmap() {return _squarBrushPixmap;}

private:

    QPixmap _defaultBrushPixmap;
    QPixmap _circleBrushPixmap;
    QPixmap _squarBrushPixmap;

	bool mIsMouseDown = false;
	bool mIsFocused = false;

    int mMousePosX = 0;
    int mMousePosY = 0;

    int mLastMousePosX = 0;
    int mLastMousePosY = 0;

    QImage* image_ = 0;

    bool _drawCursor = false;

    QCursor brushCursor;
    InteractiveMCPTPlugin* plugin = 0;

protected:
	virtual void enterEvent(QEvent* ev);
	virtual void leaveEvent(QEvent* ev);
	virtual void mouseMoveEvent(QMouseEvent* ev);
    virtual void mousePressEvent(QMouseEvent* ev);
	virtual void mouseReleaseEvent(QMouseEvent* ev);
};
