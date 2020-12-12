#include "clientconnectdlg.h"
#include "ui_clientconnectdlg.h"

ClientConnectDlg::ClientConnectDlg(Controller *controller, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ClientConnectDlg)
{
    ui->setupUi(this);
    this->controller = controller;
    connect(ui->btnConnect, &QPushButton::clicked, this, &ClientConnectDlg::onConnectClicked);
    connect(ui->btnCancel, &QPushButton::clicked, this, &ClientConnectDlg::onCancelClicked);

}

void ClientConnectDlg::onCancelClicked()
{
    controller->getClient()->cancelConnectingToHost();
    ui->btnCancel->setEnabled(false);
    ui->btnConnect->setEnabled(true);
    ui->progressBarConnecting->setMaximum(100);
    ui->progressBarConnecting->setValue(0);
}

void ClientConnectDlg::onConnectClicked()
{
    QHostAddress serverIP;
    if(!serverIP.setAddress(ui->lineEditIP->text()))
    {
        QMessageBox::warning(this, tr("Warning"), tr("Invalid IP"));
        return ;
    }
    int port = ui->lineEditPort->text().toInt();
    if(port <= 0)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Invalid Port"));
        return ;
    }
    ui->btnConnect->setEnabled(false);
    ui->btnCancel->setEnabled(true);
    ui->progressBarConnecting->setMaximum(0);
    ui->progressBarConnecting->setMinimum(0);
    connect(controller->getClient(), &TcpClient::connected, this, &ClientConnectDlg::accept);
    controller->clientConnect(serverIP, port);
}

ClientConnectDlg::~ClientConnectDlg()
{
    delete ui;
}
