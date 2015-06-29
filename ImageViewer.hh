#pragma once

#include <QLabel>


class ImageViewer : public QLabel{
    Q_OBJECT

signals:
    void mousePressed(QMouseEvent* ev);

public:

    ImageViewer(QWidget* parent = 0) : QLabel(parent){}

protected:
    virtual void mousePressEvent(QMouseEvent* ev);

};
