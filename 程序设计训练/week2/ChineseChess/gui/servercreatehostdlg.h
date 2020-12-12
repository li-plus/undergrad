#ifndef SERVERCREATEHOSTDLG_H
#define SERVERCREATEHOSTDLG_H

#include <QDialog>
#include "core/controller.h"


namespace Ui {
class ServerCreateHostDlg;
}

class ServerCreateHostDlg : public QDialog
{
    Q_OBJECT

public:
    explicit ServerCreateHostDlg(Controller *controller, QWidget *parent = nullptr);
    ~ServerCreateHostDlg();
    void onCreateClicked();
    void onCancelClicked();
private:
    Ui::ServerCreateHostDlg *ui;
    Controller *controller;
};

#endif // SERVERCREATEHOSTDLG_H
