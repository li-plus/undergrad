#ifndef CLIENTCONNECTDLG_H
#define CLIENTCONNECTDLG_H

#include <QDialog>
#include <QHostAddress>
#include <QMessageBox>
#include "core/controller.h"

namespace Ui {
class ClientConnectDlg;
}

class ClientConnectDlg : public QDialog
{
    Q_OBJECT

public:
    explicit ClientConnectDlg(Controller *controller, QWidget *parent = nullptr);
    ~ClientConnectDlg();
    void onConnectClicked();
    void onCancelClicked();


private:
    Ui::ClientConnectDlg *ui;
    Controller *controller;
};

#endif // CLIENTCONNECTDLG_H
