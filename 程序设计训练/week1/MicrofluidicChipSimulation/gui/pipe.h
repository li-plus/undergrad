#ifndef PIPE_H
#define PIPE_H

#include <QCursor>
#include <QGraphicsItem>

#include "core/coresimulator.h"
#include "gui/paintarea.h"
#include "definition.h"

class Pipe : public QObject, public QGraphicsItem
{
public:
    Pipe(int index, int row, int col, Pose pose, bool enableHidden, int totalLength,
         const PaintArea *paintArea, double displayRatio = 0.05, double flow = DEFAULT_FLOW, QObject *parent = nullptr);
    virtual ~Pipe() override;

    void updateStatus(int row, int col, double length, double flow = DEFAULT_FLOW);

    virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    virtual QPainterPath shape() const override;
    virtual QRectF boundingRect() const override;

    int col() const { return m_col; }
    int row() const { return m_row; }
    int getIndex(){ return this->index; }
    int getWidth(){ return this->pipeWidth; }
    double getFlow(){ return this->flow; }
    bool getIsHidden(){ return this->m_isHidden; }
    bool getEnableHidden(){ return this->m_enableHidden; }
    QSize sizeRealVal() const { return sizeReal; }
    double displaySizeRatio() const { return displayRatio; }

    void setHighlight(bool isHighlight);
    void setCol(int col) { m_col = col; update(); }

    Pose pose() const { return m_pose; }


protected:
    static QRectF generateShapeRect(const QPoint &center, const QSize& size);

private:
    int m_col, m_row, m_totalLength, pipeWidth, index;
    double flow;
    QSize sizeReal;
    Pose m_pose;
    bool m_isHidden, m_enableHidden;
    double displayRatio;
    const CoreSimulator *simulator;
    const PaintArea *paintArea;
    bool isHighlight;

};

#endif // PIPE_H
