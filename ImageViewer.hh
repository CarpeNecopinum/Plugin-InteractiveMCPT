#pragma once

#include <QLabel>


class ImageViewer : public QLabel{
    Q_OBJECT

signals:
    void mousePressed(QMouseEvent* ev);
	void mouseReleased(QMouseEvent* ev);
	void mouseMoved(QMouseEvent* ev);
	void mouseLeaved(QEvent* ev);
	void mouseEntered(QEvent* ev);

public:
    ImageViewer(QWidget* parent = 0) : QLabel(parent){}

	inline bool isMouseDown(){ return mIsMouseDown; }
	inline bool isFocused(){ return mIsFocused; }

private:
	bool mIsMouseDown = false;
	bool mIsFocused = false;

	float mMousePosX = 0.0f;
	float mMousePosY = 0.0f;

protected:
	virtual void enterEvent(QEvent* ev);
	virtual void leaveEvent(QEvent* ev);
	virtual void mouseMoveEvent(QMouseEvent* ev);
    virtual void mousePressEvent(QMouseEvent* ev);
	virtual void mouseReleaseEvent(QMouseEvent* ev);
};
