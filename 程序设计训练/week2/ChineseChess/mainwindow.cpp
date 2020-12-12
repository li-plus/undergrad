#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "config/gameconfig.h"
#include <QtDebug>
#include <QFileDialog>
#include <QMessageBox>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // init action
    this->initActions();

    // set window attribute
    this->setWindowTitle(tr("Chinese Chess"));
}

void MainWindow::setTimer(int value)
{
    ui->lcdTimeFamily->display(value);
}
void MainWindow::setTimerEnemy(int value)
{
    ui->lcdTimeEnemy->display(value);
}
void MainWindow::loadGame()
{
    if(!controller)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Please Create a New Game First"));
        return ;
    }
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    tr("E:/Year_1/summer/Qt_summer/week2/proj_standard"), tr("*.txt"));

    if(!fileName.isEmpty())
    {
        QFile file(fileName);
        if(!file.open(QIODevice::ReadOnly|QIODevice::Text))
        {
            QMessageBox::warning(this, tr("Warning"), tr("cannot open file") + fileName, QMessageBox::Ok);
            return;
        }
        QTextStream in(&file);
        controller->loadGame(in.readAll());
        file.close();
    }
}
void MainWindow::win()
{
    ui->btnSurrender->setEnabled(false);
}
void MainWindow::lose()
{
    ui->btnSurrender->setEnabled(false);
}
void MainWindow::newGame()
{
    // show settings dialog
    settingsDlg = new SettingsDlg(this);
    settingsDlg->show();
    if(settingsDlg->exec() == QDialog::Rejected) return ;

    // init controller
    if(controller) delete controller;
    controller = new Controller(this, this);
    connect(ui->actionSurrender, &QAction::triggered, controller, &Controller::surrender);
    connect(ui->btnSurrender, &QPushButton::clicked, controller, &Controller::surrender);

    qDebug()<<"new game";
    if(gameConfig.getAppType() == CLIENT)
    {
        clientConnectDlg = new ClientConnectDlg(controller, this);
        clientConnectDlg->show();
        if(clientConnectDlg->exec() == QDialog::Rejected) return ;
    }
    else if(gameConfig.getAppType() == SERVER)
    {
        serverCreateHostDlg = new ServerCreateHostDlg(controller, this);
        serverCreateHostDlg->show();
        if(serverCreateHostDlg->exec() == QDialog::Rejected) return;
    }

    // init other object
    if(gameView) delete gameView;
    gameView = new GameView(controller, this);

    ui->layoutMain->addWidget(gameView);
    ui->frame->setLayout(ui->layoutMain);
    // send other objects' pointers to controller
    controller->setGameView(gameView);
    controller->playMusic("NewGame");
    controller->initTimer();

    ui->btnSurrender->setEnabled(true);
}

void MainWindow::saveGame()
{
    if(!controller)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Please Create a New Game First"));
        return ;
    }
    qDebug()<<"save game";
    QString fileName;

    fileName = QFileDialog::getSaveFileName(this, tr("Save File"), tr("./abc.txt"));

    QFile file(fileName);
    if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, tr("Warning"), tr("cannot open file") + fileName);
        return ;
    }
    QTextStream out(&file);
    out << controller->getDataString();
    file.close();
}

void MainWindow::showSettingsDlg(bool exec)
{
    settingsDlg = new SettingsDlg(this);
    settingsDlg->show();
    if(exec)
    {
        settingsDlg->exec();
    }
}

void MainWindow::initActions()
{
    connect(ui->actionExit, &QAction::triggered, this, &MainWindow::close);
    connect(ui->actionNew, &QAction::triggered, this, &MainWindow::newGame);
    connect(ui->actionOpen, &QAction::triggered, this, &MainWindow::loadGame);
    connect(ui->actionSave, &QAction::triggered, this, &MainWindow::saveGame);
    connect(ui->actionAudio, &QAction::triggered, [=]
    {
        gameConfig.setAudioOn(ui->actionAudio->isChecked());
    });
    connect(ui->actionHint, &QAction::triggered, [=]
    {
        gameConfig.setShowHint(ui->actionHint->isChecked());
    });

    connect(ui->btnExit, &QPushButton::clicked, ui->actionExit, &QAction::triggered);
    connect(ui->btnLoadGame, &QPushButton::clicked, ui->actionOpen, &QAction::triggered);
    connect(ui->btnNewGame, &QPushButton::clicked, ui->actionNew, &QAction::triggered);
    connect(ui->btnSaveGame, &QPushButton::clicked, ui->actionSave, &QAction::triggered);
    connect(ui->btnSurrender, &QPushButton::clicked, ui->actionSurrender, &QAction::triggered);
}
MainWindow::~MainWindow()
{
    delete ui;
}
