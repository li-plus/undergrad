#ifndef PIPENODE_H
#define PIPENODE_H

#include <QGraphicsItem>

#include "definition.h"


class PipeNode : public QGraphicsItem
{

public:
    PipeNode(int sideLength, int row, int col, double displayRatio);
    virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    virtual QRectF boundingRect() const override;
    virtual QPainterPath shape() const override;

private:
    int sideLength;
    int row, col;
    double displayRatio;
};

#endif // PIPENODE_H
