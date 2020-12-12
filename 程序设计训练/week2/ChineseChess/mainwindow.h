#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "gui/gameview.h"
#include "core/controller.h"
#include "gui/settingsdlg.h"
#include "gui/clientconnectdlg.h"
#include "gui/servercreatehostdlg.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void initActions();
    void setTimer(int value);
    void setTimerEnemy(int value);

public slots:
    void loadGame();
    void newGame();
    void saveGame();
    void showSettingsDlg(bool exec);
    void win();
    void lose();

private:
    Ui::MainWindow *ui;
    GameView *gameView = nullptr;
    Controller *controller = nullptr;
    SettingsDlg *settingsDlg = nullptr;
    ClientConnectDlg *clientConnectDlg = nullptr;
    ServerCreateHostDlg *serverCreateHostDlg = nullptr;
};

#endif // MAINWINDOW_H
