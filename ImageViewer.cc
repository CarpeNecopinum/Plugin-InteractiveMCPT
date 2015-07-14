#include "ImageViewer.hh"

#include <QMouseEvent>
#include "InteractiveMCPT_.hh"

void ImageViewer::updateCursorIcon(){

    int brushSize = plugin->getInteractiveDrawing().getBrush().getSize();
    int size = 2 * brushSize;

    _squarBrushPixmap = QPixmap(size, size);
    _squarBrushPixmap.fill(Qt::blue);

    _circleBrushPixmap = QPixmap(size, size);
    _circleBrushPixmap.fill(Qt::transparent);


    QImage color = QImage(size,size, QImage::Format_RGBA8888);

    for (int y = 0; y < size; ++y){
        for (int x = 0; x < size; ++x){
            if (((x-brushSize) * (x-brushSize) + (y-brushSize) * (y-brushSize)) > brushSize * brushSize)
                color.setPixel(x, y, qRgba(0, 0, 0, 0));
            else
                color.setPixel(x, y, qRgba(0, 0, 255, 255));
        }
    }

    _circleBrushPixmap = QPixmap::fromImage(color);

    changeCursor((int) plugin->getInteractiveDrawing().getActiveBrush());
}

void ImageViewer::changeCursor(int type){

    QCursor myCursor;

    switch (type) {
    case 0:
        myCursor = QCursor(_defaultBrushPixmap);
        break;
    case 1:
        myCursor = QCursor(_squarBrushPixmap);
        break;
    case 2:
        myCursor = QCursor(_circleBrushPixmap);
        break;
    case 3:
        myCursor = QCursor(_circleBrushPixmap);
        break;
    default:
        myCursor = QCursor(_defaultBrushPixmap);
        break;
    }

    this->setCursor(myCursor);

}

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
