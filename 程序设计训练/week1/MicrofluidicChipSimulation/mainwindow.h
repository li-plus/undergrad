#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QVector>
#include <QGridLayout>
#include <QSignalMapper>
#include "gui/pipe.h"
#include "definition.h"
#include "core/coresimulator.h"
#include "gui/paintarea.h"
#include "gui/pipeioconfigdlg.h"
#include "gui/flowdesigndlg.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void setFlow(const vector<double> &ans);
    void changeCurFileName(const QString &fileName);
    void setSpeed(const vector<double> &speeds);
    void hideDesignDlg();


signals:
    void pipeWidthUpdated(int i, int w);


public slots:
    void showConfigIODlg(bool isNew);
    void showPipeInfo(int i, int w, double flow, bool isHidden, bool enableHidden);
    void closeProject();
    void saveFile();
    void openFile();
    void newProject();
    void designFlow();

private:
    Ui::MainWindow *ui;
    PaintArea *paintArea;
    QVector<Pipe*> pipesCentral;
    QVector<Pipe*> pipesInput;
    QVector<Pipe*> pipesOutput;
    int boardLength;

    CoreSimulator simulator;
    PipeIOConfigDlg *ioDlg;
    QString currentFileName;
    FlowDesignDlg *flowDesignDlg;
};

#endif // MAINWINDOW_H
