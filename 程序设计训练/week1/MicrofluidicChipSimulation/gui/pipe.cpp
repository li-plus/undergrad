#include "pipe.h"

#include "definition.h"


#include <QPainter>
#include <QtDebug>
#include <QMouseEvent>
#include <QGraphicsSceneEvent>

Pipe::Pipe(int index, int row, int col, Pose pose, bool enableHidden, int totalLength, const PaintArea *paintArea,
           double displayRatio, double flow, QObject *parent) : QObject (parent)
{
    this->isHighlight = false;
    this->index = index;
    m_col = col;
    m_row = row;
    m_pose = pose;

    m_isHidden = false;
    m_enableHidden = enableHidden;
    m_totalLength = totalLength;
    this->displayRatio = displayRatio;
    this->flow = flow;
    if(pose == HORIZONTAL)
    {
        this->sizeReal.setWidth(SINGLE_PIPE_LENGTH);
        this->sizeReal.setHeight(SINGLE_PIPE_WIDTH);
        this->setPos(((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * col + SINGLE_PIPE_LENGTH / 2 + SINGLE_PIPE_WIDTH) * displayRatio,
                     ((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * row + SINGLE_PIPE_WIDTH / 2 ) * displayRatio);
    }
    else if(pose == VERTICAL)
    {
        this->sizeReal.setWidth(SINGLE_PIPE_WIDTH);
        this->sizeReal.setHeight(SINGLE_PIPE_LENGTH);
        this->setPos(((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * col + SINGLE_PIPE_WIDTH / 2) * displayRatio,
                     ((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * row  + SINGLE_PIPE_LENGTH / 2 + SINGLE_PIPE_WIDTH)* displayRatio);
    }
    setCursor(QCursor(Qt::PointingHandCursor));
    this->paintArea = paintArea;

    // init pipe width
    this->pipeWidth = SINGLE_PIPE_WIDTH;

}
void Pipe::setHighlight(bool isHighlight)
{
    this->isHighlight = isHighlight;
    update();
}
QRectF Pipe::generateShapeRect(const QPoint &center, const QSize& size)
{
    return QRectF(center.x() - size.width() / 2, center.y() - size.height() / 2, size.width(), size.height());
}


QRectF Pipe::boundingRect() const
{
    return generateShapeRect(QPoint(0,0),
                             QSize(max(this->sizeReal.width(), SINGLE_PIPE_WIDTH) * displayRatio,
                                   max(this->sizeReal.height(), SINGLE_PIPE_WIDTH) * displayRatio));
}
void Pipe::updateStatus(int row, int col, double length, double flow)
{
    m_col = col;
    m_row = row;
    m_isHidden = (length == 0);
    if(length > 0) pipeWidth = 200 / length;
    this->flow = flow;
    // note that width must be within range
    if(m_pose == HORIZONTAL)
    {
        this->setPos(((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * col + SINGLE_PIPE_LENGTH / 2 + SINGLE_PIPE_WIDTH) * displayRatio,
                     ((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * row + SINGLE_PIPE_WIDTH / 2 ) * displayRatio);
        this->sizeReal.setHeight(pipeWidth);
    }
    else if(m_pose == VERTICAL)
    {
        this->setPos(((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * col + SINGLE_PIPE_WIDTH / 2) * displayRatio,
                     ((SINGLE_PIPE_LENGTH + SINGLE_PIPE_WIDTH) * row  + SINGLE_PIPE_LENGTH / 2 + SINGLE_PIPE_WIDTH)* displayRatio);
        this->sizeReal.setWidth(pipeWidth);
    }
    update();
}

QPainterPath Pipe::shape() const
{
    QPainterPath path;
    QRectF tmpRect = generateShapeRect(QPoint(0,0), QSize(this->sizeReal.width() * displayRatio,
                                                          this->sizeReal.height() * displayRatio));
    path.addRect(tmpRect);
    return path;
}

void Pipe::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QColor color;
    QPen pen = painter->pen();
    if (m_isHidden)
    {
        if(!isHighlight) return;
        else pen.setStyle(Qt::DotLine);
    }
    else
    {
        if(isHighlight)  color.setRgb(255,255,255);
        else color.setHsv(255 * ( 1.0 - flow / 400.0), 255, 255);
    }
    painter->save();

    painter->setPen(pen);
    painter->setBrush(QBrush(color));
    painter->drawPath(this->shape());

    painter->restore();
}

Pipe::~Pipe()
{

}
