#pragma once

#include <QLabel>
#include <OpenFlipper/BasePlugin/LoggingInterface.hh>

class ImageViewer : public QLabel{
    Q_OBJECT

signals:
    void mousePressed(QMouseEvent* ev);
	void mouseReleased(QMouseEvent* ev);
	void mouseMoved(QMouseEvent* ev);
	void mouseLeaved(QEvent* ev);
	void mouseEntered(QEvent* ev);

public:
	ImageViewer(QImage* image, QWidget* parent = 0) : QLabel(parent){
		this->image_ = image;
	}

	inline bool isMouseDown(){ return mIsMouseDown; }
	inline bool isFocused(){ return mIsFocused; }

	QImage* getImage(){ return image_; }

private:
	bool mIsMouseDown = false;
	bool mIsFocused = false;

    int mMousePosX = 0;
    int mMousePosY = 0;

    int mLastMousePosX = 0;
    int mLastMousePosY = 0;

	QImage* image_;

protected:
	virtual void enterEvent(QEvent* ev);
	virtual void leaveEvent(QEvent* ev);
	virtual void mouseMoveEvent(QMouseEvent* ev);
    virtual void mousePressEvent(QMouseEvent* ev);
	virtual void mouseReleaseEvent(QMouseEvent* ev);
};
