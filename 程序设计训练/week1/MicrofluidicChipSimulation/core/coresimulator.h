#ifndef CORESIMULATOR_H
#define CORESIMULATOR_H


#include <vector>
#include <cmath>
#include <QObject>
#include <QVector>
#include <QDataStream>
#include "definition.h"
#include "core/workerevolve.h"

using namespace std;

static const double NAX = 0.000000001;

class PaintArea;
class Pipe;
class MainWindow;


class AllData
{
public:
    int centralNum, sideLength;
    bool onlyOutputFlow;
    vector<double> pipeLengths;
    vector<int> width;
    vector<int> inputCol;
    vector<int> outputCol;

    AllData(){}
    AllData(int centralNum, int sideLength, bool onlyOutputFlow, vector<double> pipeLengths, vector<int> width,
            vector<int> inputCol, vector<int> outputCol) {
        this->centralNum= centralNum;
        this->sideLength = sideLength;
        this->onlyOutputFlow = onlyOutputFlow;
        this->pipeLengths = pipeLengths;
        this->width = width;
        this->inputCol = inputCol;
        this->outputCol = outputCol;
    }
    friend QDataStream &operator>>(QDataStream &in, AllData &data)
    {
        QVector<int> tmpVI;
        QVector<double> tmpVD;
        in >> data.centralNum >> data.onlyOutputFlow >> data.sideLength;
        in>> tmpVI;
        data.inputCol = tmpVI.toStdVector();
        in>> tmpVI;
        data.outputCol = tmpVI.toStdVector();
        in >> tmpVI;
        data.width = tmpVI.toStdVector();
        in>>tmpVD;
        data.pipeLengths = tmpVD.toStdVector();
        return in;
    }
    friend QDataStream &operator<<(QDataStream &out, const AllData &data)
    {
        out<< data.centralNum<< data.onlyOutputFlow<<data.sideLength<< QVector<int>::fromStdVector(data.inputCol)
           <<QVector<int>::fromStdVector(data.outputCol)<<QVector<int>::fromStdVector(data.width)
          <<QVector<double>::fromStdVector(data.pipeLengths);
        return out;
    }
};


class CoreSimulator : public QObject{


public:
    CoreSimulator(MainWindow *mainWindow, PaintArea *paintArea, QObject *parent = nullptr);
    void assign(int sideLength, const vector<int> &inputCol, const vector<int> &outputCol, const vector<double> &pipeLength,
                const vector<int> &width);
    void init(int sideLength, const vector<int> &inputCol, const vector<int> &outputCol, bool initCentral = true);
    vector<double> calculateFlow();
    vector<double> calculateSpeed(const vector<double> &flows);
    void updatePipeVisible(int index, bool isHidden);
    void configPipeIO(int sideLength, vector<int> input, vector<int> output, bool isNew);
    int sideLenght() const { return this->m_sideLength; }

    const vector<int> &inputColumn() const { return this->inputCol; }
    const vector<int> &outputColumn() const { return this->outputCol; }
    const vector<int> &pipesWidth() const { return this->m_width; }


    AllData getAllData();
    void loadAllData(const AllData &data);

    vector<double> outputFlows(const vector<double> &ans) const;
    // helper functions
    static int posToIndex(int row, int col, Pose pose, int sideLength);
    static Pose indexToPos(int i, int totalLength, QPoint &posAns);
    int leftPipeIndex(int i);
    int rightPipeIndex(int i);
    int upPipeIndex(int i);
    int downPipeIndex(int i);


public slots:
    void updatePipeWidth(int i, int w);
    void designFlow(vector<double> targetFlow);
    void startThreadEvolve(vector<double> targetFlow);
    void providePipeInfo(int index);
    void stopThreadImmediately();

private:
    int centralNum, m_sideLength;
    bool onlyOutputFlow;
    vector<double> m_pipeLengths;
    vector<int> m_width;
    vector<int> inputCol, outputCol;

    PaintArea *paintArea;
    MainWindow *mainWindow;

    WorkerEvolve *workerEvolve;
};



#endif // CORESIMULATOR_H
