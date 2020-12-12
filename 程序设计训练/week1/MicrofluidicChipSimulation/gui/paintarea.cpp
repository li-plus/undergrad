#include "paintarea.h"

#include "definition.h"
#include "gui/pipe.h"

#include <QtDebug>

#include <QMouseEvent>


PaintArea::PaintArea(QWidget *parent) : QGraphicsView (parent)
{

}
void PaintArea::init(const CoreSimulator *sim, int sideLength)
{
    // no scroll bar
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    // init variables
    this->pipeChosen = nullptr;
    this->isPressed = false;
    this->displayRatio = 0.05;
    this->sideLength = sideLength;
    this->centralNum = 2 * sideLength * sideLength - 2 * sideLength;
    this->simulator = sim;
    // init scene
    qDebug()<<"init scene";
    scene.clear();
    scene.setSceneRect(-2000,-2000,4500,4500);
    this->setScene(&scene);

    // init pipes
    pipesInput = QVector<Pipe*>(INPUT_NUM);
    pipesOutput = QVector<Pipe*>(OUTPUT_NUM);
    pipesCentral = QVector<Pipe*>(centralNum);
    textItemAnswer = QVector<QGraphicsTextItem*>(OUTPUT_NUM);
    textItemInput = QVector<QGraphicsTextItem*> (INPUT_NUM);
    for(int i=0; i<INPUT_NUM; i++)
    {
        pipesInput[i] = new Pipe(centralNum + i, -1, i, VERTICAL, false, sideLength, this, 0.05, DEFAULT_FLOW, this);
        scene.addItem(pipesInput[i]);
    }
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        pipesOutput[i] = new Pipe(centralNum + INPUT_NUM + i, sideLength - 1, i, VERTICAL, false, sideLength, this, 0.05,
                                  DEFAULT_FLOW, this);
        scene.addItem(pipesOutput[i]);
    }
    for(int i=0; i<centralNum; i++)
    {
        QPoint position;
        Pose pose = CoreSimulator::indexToPos(i, sideLength, position);
        pipesCentral[i] = new Pipe(i, position.x(), position.y(), pose, true, sideLength, this, 0.05, DEFAULT_FLOW, this);
        scene.addItem(pipesCentral[i]);
    }
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        textItemAnswer[i] = new QGraphicsTextItem;
        scene.addItem(textItemAnswer[i]);
    }
    for(int i=0; i<INPUT_NUM; i++)
    {
        textItemInput[i] = new QGraphicsTextItem;
        scene.addItem(textItemInput[i]);
    }
    // init nodes
    for(int i=0; i<sideLength; i++)
    {
        for(int j=0; j<sideLength; j++)
        {
            scene.addItem(new PipeNode(sideLength, i, j, displayRatio));
        }
    }
    // init answers

}

void PaintArea::paint(int sideLength, const std::vector<double>& pipeLengths,
                      const std::vector<int> &inputCol, const std::vector<int> &outputCol,
                      const std::vector<double> &answer)
{
    this->show();
    this->sideLength = sideLength;
    centralNum = 2 * sideLength * sideLength - 2 * sideLength;
    bool onlyOutputFlow = (answer.size() == OUTPUT_NUM);

    for(int i=0; i<centralNum; i++)
    {
        double tmpOutputFlow = DEFAULT_FLOW;
        if(!onlyOutputFlow) tmpOutputFlow = answer[i];

        pipesCentral[i]->updateStatus(pipesCentral[i]->row(), pipesCentral[i]->col(), pipeLengths[i], tmpOutputFlow);
    }
    for(int i=0; i<INPUT_NUM; i++)
    {
        double tmpOutputFlow = DEFAULT_FLOW;
        if(!onlyOutputFlow) tmpOutputFlow = answer[centralNum + i];

        pipesInput[i]->updateStatus(pipesInput[i]->row(), inputCol[i], pipeLengths[centralNum + i], tmpOutputFlow);

        textItemInput[i]->setPlainText(QString::number(DEFAULT_FLOW));
        textItemInput[i]->setPos(pipesInput[i]->pos().x() - SINGLE_PIPE_WIDTH * pipesInput[i]->displaySizeRatio() - 10,
                                  pipesInput[i]->pos().y() - SINGLE_PIPE_LENGTH * pipesInput[i]->displaySizeRatio() + 5);

    }
    for(int i=0; i<OUTPUT_NUM; i++)
    {
        double tmpOutputFlow = DEFAULT_FLOW;
        if(!onlyOutputFlow) tmpOutputFlow = answer[centralNum + INPUT_NUM + i];

        pipesOutput[i]->updateStatus(pipesOutput[i]->row(), outputCol[i], pipeLengths[centralNum + INPUT_NUM + i],
                tmpOutputFlow);

        textItemAnswer[i]->setPlainText(QString::number(tmpOutputFlow, 'g', 3));
        textItemAnswer[i]->setPos(pipesOutput[i]->pos().x() - SINGLE_PIPE_WIDTH * pipesOutput[i]->displaySizeRatio() - 10,
                                  pipesOutput[i]->pos().y() + SINGLE_PIPE_LENGTH * pipesOutput[i]->displaySizeRatio() / 2);
    }
    scene.update();
}
void PaintArea::wheelEvent(QWheelEvent *event)
{
    double factor;
    if(event->delta() < 0) factor = 0.9;
    else factor = 1.1;

    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

    scale(factor, factor);
    setTransformationAnchor(QGraphicsView::AnchorViewCenter);

    scene.update();

}

void PaintArea::mousePressEvent(QMouseEvent *event)
{
    QPointF pos = this->mapToScene(event->pos());
    // device transform is the transformation between scene and view
    if(QGraphicsItem *item = scene.itemAt(pos, QTransform()))
    {
        if(Pipe* pipeItem = dynamic_cast<Pipe*>(item))
        {
            if(pipeChosen) pipeChosen->setHighlight(false);
            pipeItem->setHighlight(true);

            qDebug()<<"pipe selected";
            this->pipeChosen = pipeItem;
            emit chosenPipe(pipeItem->getIndex());
        }
    }
    else
    {
        isPressed = true;
        oldPoint = this->mapToScene(event->pos());

        if(this->pipeChosen) this->pipeChosen->setHighlight(false);
        this->pipeChosen = nullptr;
        emit chosenPipe(-1);
    }
    scene.update();
}

PaintArea::~PaintArea()
{

}
