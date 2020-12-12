#include "servercreatehostdlg.h"
#include "ui_servercreatehostdlg.h"

#include <QMessageBox>


ServerCreateHostDlg::ServerCreateHostDlg(Controller *controller, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ServerCreateHostDlg)
{
    ui->setupUi(this);
    ui->lineEditIP->setText(QHostAddress(QHostAddress::LocalHost).toString() );
    this->controller = controller;


    connect(ui->btnCreate, &QPushButton::clicked, this, &ServerCreateHostDlg::onCreateClicked);
    connect(ui->btnCancel, &QPushButton::clicked, this, &ServerCreateHostDlg::onCancelClicked);
}
void ServerCreateHostDlg::onCancelClicked()
{
    controller->getServer()->close();
    ui->progressBarConnect->setMaximum(100);
    ui->btnCancel->setEnabled(false);
    ui->btnCreate->setEnabled(true);
}
void ServerCreateHostDlg::onCreateClicked()
{

    int port = ui->lineEditPort->text().toInt();
    if(port <=0 || port > 65535)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Invalid IP"));
        return ;
    }
    if(controller->serverCreateHost(port))
    {
        // pass
    }
    else
    {
        QMessageBox::warning(this, tr("Warning"), tr("Cannot Create Host"));
    }
    connect(controller->getServer(), &TcpServer::twoClientConnected, this, &ServerCreateHostDlg::accept);

    ui->progressBarConnect->setMaximum(0);
    ui->btnCancel->setEnabled(true);
    ui->btnCreate->setEnabled(false);


}
ServerCreateHostDlg::~ServerCreateHostDlg()
{
    delete ui;
}
