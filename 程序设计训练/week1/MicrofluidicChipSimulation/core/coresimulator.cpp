#include "coresimulator.h"
#include "gui/paintarea.h"
#include "mainwindow.h"
#include "core/simulatorapi.h"
#include <stdlib.h>
#include <QtDebug>
#include <exception>


CoreSimulator::CoreSimulator(MainWindow *mainWindow, PaintArea *paintArea, QObject *parent)
    : QObject (parent)
{
    this->init(5, vector<int>({0,1}), vector<int>({0,1,2}), true);
    // init paint area
    this->onlyOutputFlow = false;
    this->paintArea = paintArea;
    paintArea->init(this, m_sideLength);
    // init main window
    this->mainWindow = mainWindow;

    connect(mainWindow, &MainWindow::pipeWidthUpdated, this, &CoreSimulator::updatePipeWidth);
    connect(paintArea, &PaintArea::chosenPipe, this, &CoreSimulator::providePipeInfo);
    qDebug()<<m_pipeLengths;
}
void CoreSimulator::assign(int sideLength, const vector<int> &inputCol, const vector<int> &outputCol,
                           const vector<double> &pipeLength, const vector<int> &width)
{
    this->m_sideLength = sideLength;
    this->inputCol = inputCol;
    this->outputCol = outputCol;
    this->m_pipeLengths = pipeLength;
    this->m_width = width;
}
void CoreSimulator::init(int sideLength, const vector<int> &inputCol, const vector<int> &outputCol,
                         bool initCentral)
{
    this->m_sideLength = sideLength;
    this->inputCol = inputCol;
    this->outputCol = outputCol;
    this->centralNum = 2 * m_sideLength * m_sideLength - 2 * m_sideLength;
    this->m_width = vector<int> (centralNum + INPUT_NUM + OUTPUT_NUM, SINGLE_PIPE_WIDTH);
    // randomize
    if(initCentral)
    {
        this->m_pipeLengths = vector<double> (centralNum + INPUT_NUM + OUTPUT_NUM, 1.0);
        for(int i = 0; i<centralNum; i++)
        {
            if(rand() % 10 > 8)
            {
                m_pipeLengths[i] = 0;
            }
        }
    }
}

void CoreSimulator::providePipeInfo(int i)
{
    vector<double> flows(calculateFlow());
    if(i < 0)
    {
        mainWindow->showPipeInfo(-1,0,0,0,0);
    }
    else
    {
        mainWindow->showPipeInfo(i, m_width[i], flows[i], m_pipeLengths[i] == 0, i < centralNum);
    }
}
void CoreSimulator::configPipeIO(int sideLength, std::vector<int> input, std::vector<int> output, bool isNew)
{
    qDebug()<<"input"<<inputCol<<"output "<<outputCol;
    if(this->m_sideLength == sideLength && !isNew)
    {
        qDebug()<<"resize board";
        this->init(sideLength, input, output, false);
    }
    else
    {
        this->init(sideLength, input, output, true);
    }
    paintArea->init(this, sideLength);
    vector<double> flows(calculateFlow());
    paintArea->paint(m_sideLength, m_pipeLengths, input, output, flows);

    mainWindow->setFlow(outputFlows(flows));
    mainWindow->setSpeed(calculateSpeed(flows));
}
vector<double> CoreSimulator::calculateFlow()
{
    // calculate Flow according to private variable onlyOutputFlow
    qDebug()<<"calculating"<<m_sideLength<<m_pipeLengths<<inputCol<<outputCol;
    return caluconspeed(m_sideLength, m_pipeLengths, inputCol[0], inputCol[1],
            outputCol[0], outputCol[1], outputCol[2]);
}
vector<double> CoreSimulator::outputFlows(const vector<double> &ans) const
{
    if(ans.size() == OUTPUT_NUM)
    {
        return ans;
    }
    else
    {
        return vector<double>(ans.end() - OUTPUT_NUM, ans.end());
    }
}
void CoreSimulator::updatePipeVisible(int index, bool isHidden)
{
    if(isHidden)
        this->m_pipeLengths[index] = 0;
    else
        this->m_pipeLengths[index] = 200.0 / m_width[index];
    vector<double> flows(calculateFlow());
    paintArea->paint(m_sideLength, m_pipeLengths, inputCol, outputCol, flows);
    mainWindow->setFlow(outputFlows(flows));
    mainWindow->setSpeed(calculateSpeed(flows));
}
void CoreSimulator::updatePipeWidth(int i, int w)
{
    m_width[i] = w;
    m_pipeLengths[i] = SINGLE_PIPE_WIDTH / (double)w;
    vector<double> flows(calculateFlow());
    paintArea->paint(m_sideLength, m_pipeLengths, inputCol, outputCol, flows);
    mainWindow->setFlow(outputFlows(flows));
    mainWindow->setSpeed(calculateSpeed(flows));
    mainWindow->showPipeInfo(i, w, flows[i], m_pipeLengths[i] == 0, i < centralNum);
}
vector<double> CoreSimulator::calculateSpeed(const vector<double> &flows)
{
    vector<double> speeds(flows.size());
    for(auto i=0;i<speeds.size(); i++)
    {
        if(m_width[i]) speeds[i] = flows[i] / m_width[i];
        else qDebug()<<"error:: divided by zero";
    }
    return speeds;
}

void CoreSimulator::designFlow(vector<double> lengths)
{

    this->init(m_sideLength, inputCol, outputCol, true);
    qDebug()<<"received flow";

    this->m_pipeLengths = lengths;


    this->workerEvolve->stopImmediately();
    this->workerEvolve->wait();

    mainWindow->hideDesignDlg();

    for(int i=0;i<centralNum + INPUT_NUM + OUTPUT_NUM;i++)
    {
        m_width[i] = m_pipeLengths[i] ? 200.0 / m_pipeLengths[i] : SINGLE_PIPE_WIDTH;
    }
    paintArea->init(this, m_sideLength);
    vector<double> flow(calculateFlow());
    paintArea->paint(m_sideLength, m_pipeLengths, inputCol, outputCol, flow);
    mainWindow->setFlow(flow);
    mainWindow->setSpeed(calculateSpeed(flow));
}
void CoreSimulator::startThreadEvolve(vector<double> targetFlow)
{
    workerEvolve = new WorkerEvolve(this);
    connect(workerEvolve, &WorkerEvolve::lengthsFound, this, &CoreSimulator::designFlow, Qt::QueuedConnection);
    workerEvolve->init(targetFlow, m_sideLength, inputCol, outputCol);
    workerEvolve->start();
}
void CoreSimulator::stopThreadImmediately()
{
    if(workerEvolve)
    {
         workerEvolve->stopImmediately();
         workerEvolve->wait();
    }

}
AllData CoreSimulator::getAllData()
{
    return AllData(centralNum, m_sideLength, onlyOutputFlow, m_pipeLengths, m_width, inputCol, outputCol);
}
void CoreSimulator::loadAllData(const AllData &data)
{
    this->centralNum = data.centralNum;
    this->inputCol =data.inputCol;
    this->onlyOutputFlow = data.onlyOutputFlow;
    this->outputCol =data.outputCol;
    this->m_pipeLengths =data.pipeLengths;
    this->m_sideLength =data.sideLength;
    this->m_width = data.width;
    qDebug()<<"loading";
    vector<double> ans(calculateFlow());
    paintArea->init(this, data.sideLength);
    paintArea->paint(m_sideLength, m_pipeLengths, inputCol, outputCol, ans);
    mainWindow->setFlow(outputFlows(ans));
}
int CoreSimulator::posToIndex(int row, int col, Pose pose, int sideLength)
{
    switch (pose) {
    case HORIZONTAL:
        return col * sideLength + row + sideLength * (sideLength - 1);
    case VERTICAL:
        return col * (sideLength - 1) + row;
    }
}
Pose CoreSimulator::indexToPos(int i, int sideLength, QPoint &posAns)
{
    if(i < sideLength * (sideLength - 1))
    {
        // vertical
        posAns = QPoint(i % (sideLength - 1), i / (sideLength - 1));
        return VERTICAL;
    }
    else
    {
        int new_i = i - sideLength * (sideLength - 1);
        posAns = QPoint(new_i % sideLength, new_i / sideLength);
        return HORIZONTAL;
    }
}
int CoreSimulator::leftPipeIndex(int i)
{
    int sideLength = m_sideLength;
    if(i < sideLength * (sideLength - 1) && i >= sideLength - 1)
    {
        return i - sideLength + 1;
    }
    else
    {
        return -1;
    }
}
int CoreSimulator::rightPipeIndex(int i)
{
    int sideLength = m_sideLength;
    if(i < sideLength * (sideLength - 1) && i <= sideLength * (sideLength - 1) - sideLength + 1)
    {
        return i + sideLength - 1;
    }
    else
    {
        return -1;
    }
}
int CoreSimulator::upPipeIndex(int i)
{
    if(i >= centralNum ) return -1;
    int sideLength = m_sideLength;
    if(i >= sideLength * (sideLength - 1) && (i - sideLength * (sideLength - 1)) % sideLength)
    {
        return i - 1;
    }
    else
    {
        return -1;
    }
}

int CoreSimulator::downPipeIndex(int i)
{
    if(i >= centralNum ) return -1;
    int sideLength = m_sideLength;
    if(i >= sideLength * (sideLength - 1) && (i - sideLength * (sideLength - 1) + 1) % sideLength)
    {
        return i + 1;
    }
    else
    {
        return -1;
    }
}
