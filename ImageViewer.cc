#include "ImageViewer.hh"

#include <QMouseEvent>

void ImageViewer::mousePressEvent(QMouseEvent* ev){
    emit mousePressed(ev);
}
