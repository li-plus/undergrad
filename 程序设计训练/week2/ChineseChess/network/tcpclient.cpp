#include "tcpclient.h"

#include <QHostAddress>
#include <QMessageBox>
#include <QDataStream>

#include "definition.h"

TcpClient::TcpClient(QObject *parent) : QTcpSocket (parent)
{
    connect(this, &QTcpSocket::readyRead, this, &TcpClient::slotReadyRead);
    connect(this, &TcpClient::disconnected, this, &TcpClient::slotDisconnected);

}

void TcpClient::keepConnectingToHost(const QHostAddress &address, quint16 port, OpenMode mode)
{
    connect(this, &TcpClient::connected, &timer, &QTimer::stop);

    connectToHost(address, port, mode);
    if(!waitForConnected(1))
    {
        qDebug()<<"waiting for connection";
        if(!timer.isActive())
        {
            timer.start(3000);
            connect(&timer, &QTimer::timeout, this, [=]
            {
                this->keepConnectingToHost(address, port, mode);
            });
        }
    }
}

void TcpClient::sendSue()
{
    QByteArray msg;
    QDataStream out(&msg, QIODevice::WriteOnly);
    out << (int)SUE;
    this->write(msg);
}

void TcpClient::cancelConnectingToHost()
{
    this->timer.stop();
}

void TcpClient::slotReadyRead()
{
    QDataStream in(this);

    while(bytesAvailable() >= sizeof(qint64))
    {
        qint64 recvlength = bytesAvailable();
        qint64 realLength;
        in >> realLength;

        if(realLength != recvlength)
        {
            qDebug()<<"tcp error"<<"real length"<<realLength<<"recvLength"<<recvlength;
            return;
        }
        qDebug()<<"successfully received";
        QByteArray msg = read(realLength - sizeof (qint64));
        emit dataReceived(msg, realLength);
    }
}

void TcpClient::slotDisconnected()
{
    emit disconnected(this->socketDescriptor());
}


