#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QtDebug>
#include <QFileDialog>
#include <QVector>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    paintArea(new PaintArea(this)),
    simulator(this, paintArea, this)
{
    qApp->setFont(QFont("Calibri"));
    ui->setupUi(this);

    this->setMinimumSize(this->size());
    QHBoxLayout *layoutL = new QHBoxLayout;
    layoutL->addWidget(paintArea);
    ui->frameL->setLayout(layoutL);

    ui->centralWidget->setLayout(ui->layoutMain);

    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    // window
    setWindowTitle(tr("Microfluidic Chip Simulator"));
    // connect
    connect(ui->actionPipeIO, &QAction::triggered, this, &MainWindow::showConfigIODlg);
    connect(ui->actionNew, &QAction::triggered, this, &MainWindow::newProject);
    connect(ui->actionClose, &QAction::triggered, this, &MainWindow::closeProject);
    connect(ui->actionSave, &QAction::triggered, this, &MainWindow::saveFile);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::openFile);
    connect(ui->actionFlowDesign, &QAction::triggered, this, &MainWindow::designFlow);

    connect(ui->sliderWidth, &QSlider::valueChanged, ui->spinBoxWidth, &QSpinBox::setValue);
    connect(ui->spinBoxWidth, QOverload<int>::of(&QSpinBox::valueChanged), ui->sliderWidth, &QSlider::setValue);
    connect(ui->sliderWidth, &QSlider::valueChanged, this, [=]{
        emit this->pipeWidthUpdated(ui->labelIndexNum->text().toInt(), ui->spinBoxWidth->value());
    });
    connect(ui->btnDelete, &QPushButton::clicked, &simulator, [=]{
        ui->btnDelete->setEnabled(false);
        ui->btnInsert->setEnabled(true);
        ui->sliderWidth->setEnabled(false);
        ui->spinBoxWidth->setEnabled(false);
        simulator.updatePipeVisible(ui->labelIndexNum->text().toInt(), true);
    });
    connect(ui->btnInsert, &QPushButton::clicked, &simulator, [=]{
        ui->btnDelete->setEnabled(true);
        ui->btnInsert->setEnabled(false);
        ui->sliderWidth->setEnabled(true);
        ui->spinBoxWidth->setEnabled(true);
        simulator.updatePipeVisible(ui->labelIndexNum->text().toInt(), false);
    });
    // config new project
    this->showConfigIODlg(true);
    // hide all first
    paintArea->hide();
    ui->frameR->hide();
}
void MainWindow::designFlow()
{
    flowDesignDlg = new FlowDesignDlg(this);

    connect(flowDesignDlg, &FlowDesignDlg::flowDesign, &simulator, &CoreSimulator::startThreadEvolve);
    connect(flowDesignDlg, &FlowDesignDlg::stopDesign, &simulator, &CoreSimulator::stopThreadImmediately);
    flowDesignDlg->show();
}
void MainWindow::saveFile()
{
    QString fileName;
    if(currentFileName.isEmpty())
    {
        fileName = QFileDialog::getSaveFileName(this, tr("Save File"), tr("./1.sim"), tr("*.sim"));
        changeCurFileName(fileName);
    }else
    {
        fileName = currentFileName;
    }
    QFile file(fileName);
    if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, tr("Warning"), tr("cannot open file") + currentFileName);
        return ;
    }
    QDataStream out(&file);
    AllData data = simulator.getAllData();
    out<<data;

    file.close();
}
void MainWindow::hideDesignDlg()
{
    if(flowDesignDlg)
    {
        flowDesignDlg->hide();
    }
}

void MainWindow::openFile()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), tr("./"), tr("*.sim"));
    if(!fileName.isEmpty())
    {
        changeCurFileName(fileName);
        QFile file(currentFileName);
        if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
        {
            QMessageBox::warning(this, tr("Warning"), tr("cannot open file") + currentFileName, QMessageBox::Ok);
            return;
        }
        QDataStream in(&file);
        AllData data;
        in>>data;

        file.close();
        simulator.loadAllData(data);

        paintArea->show();
        ui->frameR->show();
    }
}
void MainWindow::changeCurFileName(const QString & fileName)
{
    currentFileName = fileName;
    qDebug()<<"change cur filename"<<fileName;
    if(fileName.isEmpty())
    {
        setWindowTitle(tr("Microfluidic Chip Simulator"));
    }
    else
    {
        setWindowTitle(currentFileName);
    }
}

void MainWindow::closeProject()
{
    paintArea->hide();
    ui->frameR->hide();

    changeCurFileName(QString());
}
void MainWindow::setFlow(const vector<double> &ans)
{
    int size = ans.size();
    ui->lcdFlow1->display(ans[size - 3]);
    ui->lcdFlow2->display(ans[size - 2]);
    ui->lcdFlow3->display(ans[size - 1]);
}
void MainWindow::newProject()
{
    changeCurFileName(QString(""));
    paintArea->hide();
    ui->frameR->hide();
    showConfigIODlg(true);
}
void MainWindow::showConfigIODlg(bool isNew)
{
    ioDlg = new PipeIOConfigDlg(isNew, this);
    ioDlg->setIO(simulator.sideLenght(), simulator.inputColumn(), simulator.outputColumn());
    connect(ioDlg, &PipeIOConfigDlg::PipeIOChanged, &simulator, &CoreSimulator::configPipeIO);
    connect(ioDlg, &PipeIOConfigDlg::PipeIOChanged, this, [=]{ ui->frameR->show(); });
    ioDlg->show();
}
void MainWindow::showPipeInfo(int index, int width, double flow, bool isHidden, bool enableHidden)
{
    if(index<0)
    {
        ui->labelIndexNum->setText(QString());
        ui->btnDelete->setEnabled(false);
        ui->btnInsert->setEnabled(false);
        ui->spinBoxWidth->setEnabled(false);
        ui->sliderWidth->setEnabled(false);
        ui->lcdPipeFlow->display(0);
        return ;
    }
    if(isHidden)
    {
        ui->btnDelete->setEnabled(false);
        ui->btnInsert->setEnabled(true);
        ui->sliderWidth->setEnabled(false);
        ui->spinBoxWidth->setEnabled(false);
    }
    else if(enableHidden)
    {
        ui->btnDelete->setEnabled(true);
        ui->btnInsert->setEnabled(false);
        ui->sliderWidth->setEnabled(true);
        ui->spinBoxWidth->setEnabled(true);
    }
    else
    {
        ui->btnDelete->setEnabled(false);
        ui->btnInsert->setEnabled(false);
        ui->sliderWidth->setEnabled(true);
        ui->spinBoxWidth->setEnabled(true);
    }
    ui->labelIndexNum->setText(QString::number(index));
    ui->spinBoxWidth->setValue(width);
    ui->lcdPipeFlow->display(flow);
    int tmpIndex, maxWidth=3000;
    vector<int> possibleWidth;
    if((tmpIndex = simulator.leftPipeIndex(index)) > 0 && maxWidth > SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex))
    {
        maxWidth = SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex);
    }
    if((tmpIndex = simulator.rightPipeIndex(index)) > 0 && maxWidth > SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex))
    {
        maxWidth = SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex);
    }
    if((tmpIndex = simulator.upPipeIndex(index)) > 0 && maxWidth > SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex))
    {
        maxWidth = SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex);
    }
    if((tmpIndex = simulator.downPipeIndex(index)) > 0 && maxWidth > SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex))
    {
        maxWidth = SINGLE_PIPE_LENGTH * 2 - simulator.pipesWidth().at(tmpIndex);
    }
    ui->spinBoxWidth->setMaximum(maxWidth);
    ui->sliderWidth->setMaximum(maxWidth);
}
void MainWindow::setSpeed(const vector<double> &speeds)
{
    int size = speeds.size();
    ui->lcdSpeed1->display(speeds[size - 3]);
    ui->lcdSpeed2->display(speeds[size - 2]);
    ui->lcdSpeed3->display(speeds[size - 1]);
}
MainWindow::~MainWindow()
{
    delete ui;
}
