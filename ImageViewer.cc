#include "ImageViewer.hh"

#include <QMouseEvent>
#include "InteractiveMCPT_.hh"

void ImageViewer::mousePressEvent(QMouseEvent* ev){
	mIsMouseDown = true;
    emit mousePressed(ev);
}

void ImageViewer::mouseReleaseEvent(QMouseEvent* ev){
	mIsMouseDown = false;
	emit mouseReleased(ev);
}

void ImageViewer::mouseMoveEvent(QMouseEvent* ev){
    mLastMousePosX = mMousePosX;
    mLastMousePosY = mMousePosY;

	mMousePosX = ev->pos().x();
	mMousePosY = ev->pos().y();
	emit mouseMoved(ev);
}

void ImageViewer::enterEvent(QEvent* ev){
	mIsFocused = true;
	emit mouseEntered(ev);
}
void ImageViewer::leaveEvent(QEvent* ev){
	mIsFocused = false;
	emit mouseLeaved(ev);
}
