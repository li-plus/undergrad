#ifndef WORKEREVOLVE_H
#define WORKEREVOLVE_H

#include <QObject>
#include <QThread>


using namespace std;

class WorkerEvolve : public QThread
{
    Q_OBJECT
public:
    WorkerEvolve(QObject *parent = nullptr);
    void stopImmediately();
    void init(const vector<double> &targetFlow, int n, const vector<int> &inputCol, const vector<int> &outputCol);

signals:
    void lengthsFound(vector<double> lengths);

protected:
    virtual void run() override;



private:
    bool isStop;
    vector<double> targetFlow;
    vector<int> inputCol, outputCol;
    int n;

};

#endif // WORKEREVOLVE_H
