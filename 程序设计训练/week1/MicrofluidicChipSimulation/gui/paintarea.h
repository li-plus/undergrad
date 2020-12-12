#ifndef PAINTAREA_H
#define PAINTAREA_H

#include <QWidget>

#include <QGraphicsView>
#include <QGraphicsTextItem>
#include <QVector>
#include "gui/pipenode.h"
#include "core/coresimulator.h"


class Pipe;

class PaintArea : public QGraphicsView
{
    Q_OBJECT
public:
    explicit PaintArea(QWidget *parent = nullptr);
    void paint(int sideLength, const std::vector<double>& pipeLengths,
               const std::vector<int> &inputCol, const std::vector<int> &outputCol,
               const std::vector<double> &answer);
    void init(const CoreSimulator *sim, int sideLength);

    virtual ~PaintArea() override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;

private:
    QGraphicsScene scene;
    QVector<Pipe*> pipesCentral, pipesInput, pipesOutput;

    const CoreSimulator *simulator;
    QVector<QGraphicsTextItem*> textItemAnswer, textItemInput;
    int centralNum, sideLength;
    double displayRatio;
    bool isPressed;
    QPointF oldPoint;

    QTransform transform;
    Pipe* pipeChosen;

signals:
    void chosenPipe(int index);
};

#endif // PAINTAREA_H
