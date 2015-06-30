#include "ImageViewer.hh"

#include <QMouseEvent>

void ImageViewer::mousePressEvent(QMouseEvent* ev){
    emit mousePressed(ev);
	mIsMouseDown = true;
}

void ImageViewer::mouseReleaseEvent(QMouseEvent* ev){
	emit mouseReleased(ev);
	mIsMouseDown = false;
}

void ImageViewer::mouseMoveEvent(QMouseEvent* ev){
	emit mouseMoved(ev);
	mMousePosX = ev->pos().x();
	mMousePosY = ev->pos().y();
}

void ImageViewer::enterEvent(QEvent* ev){
	emit mouseEntered(ev);
	mIsFocused = true;
}
void ImageViewer::leaveEvent(QEvent* ev){
	emit mouseLeaved(ev);
	mIsFocused = false;
}