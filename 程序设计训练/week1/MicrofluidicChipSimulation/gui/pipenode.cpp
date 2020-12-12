#include "pipenode.h"

#include <QPainter>
PipeNode::PipeNode(int sideLength, int row, int col, double displayRatio)
{
    this->sideLength = sideLength;
    this->displayRatio = displayRatio;
    this->row = row;
    this->col = col;
    this->setPos((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * row * displayRatio,
                 (SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * col * displayRatio);
}
void PipeNode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    painter->save();
    painter->setBrush(QBrush(Qt::magenta));
    painter->drawPath(shape());
    painter->restore();
}
QRectF PipeNode::boundingRect() const
{
    return QRectF(0,0,0,0);
}
QPainterPath PipeNode::shape() const
{
    QPainterPath path;
    path.addRect(0,0,SINGLE_PIPE_WIDTH * displayRatio, SINGLE_PIPE_WIDTH * displayRatio);
    return path;
}
