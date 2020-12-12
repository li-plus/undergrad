#include "tcpserver.h"
#include <QDataStream>


TcpServer::TcpServer(QObject *parent) : QTcpServer (parent)
{
}

TcpServer::TcpServer(int port, QObject *parent) : QTcpServer (parent)
{
    // listen to any IPv4 address through given port
    listen(QHostAddress::Any, port);

    if(this->tcpClientList.size() >= 2)
    {
        emit twoClientConnected();
    }
    qDebug()<<"listen to port"<<port;
}

void TcpServer::reply(qintptr handle, QByteArray msg, int length)
{
    // broadcast any message from any client
    qDebug()<<"handle input"<<handle;
    for(auto it = tcpClientList.begin(); it != tcpClientList.end(); it++)
    {
        qDebug()<<"every socket descriptor"<<(*it)->socketDescriptor();
        if((*it)->socketDescriptor() == handle) continue;
        if((*it)->write(msg) != length)
        {
            continue;
        }
    }
}

void TcpServer::updateServer(QByteArray msg, int length)
{
    qDebug()<<"In server, updateServer::"<<msg;
    emit updated(msg, length);
}
void TcpServer::slotDisconnected(int descriptor)
{
    for(auto it = tcpClientList.begin(); it < tcpClientList.end(); it++)
    {
        if((*it)->socketDescriptor() == descriptor)
        {
            (*it)->deleteLater();
            tcpClientList.erase(it);
            return;
        }
    }
}
void TcpServer::receiveBytes(QTcpSocket *tcpClient)
{
    qint64 length;
    QByteArray msg;
    // decode message
    while(tcpClient->bytesAvailable() > 0)
    {
        length = tcpClient->bytesAvailable();
        msg = tcpClient->read(length);

        reply(tcpClient->socketDescriptor(), msg, length);
    }
    qDebug()<<"server receive bytes";
    // reply to every client except sender(tcpClient)
}
void TcpServer::incomingConnection(qintptr handle)
{
    qDebug()<<"incoming connection";
    QTcpSocket *tcpClient = new QTcpSocket(this);
    connect(tcpClient, &QTcpSocket::readyRead, [=]
    {
        this->receiveBytes(tcpClient);
    });

    connect(tcpClient, &QTcpSocket::disconnected, [=]
    {
        this->slotDisconnected(tcpClient->socketDescriptor());
    });
    tcpClient->setSocketDescriptor(handle);
    tcpClientList.append(tcpClient);
    qDebug()<<"connected"<<tcpClientList;
    qDebug()<<"incoming connection ip"<<tcpClient->peerAddress()<<"port"<<tcpClient->peerPort();

    if(tcpClientList.size() >= 2 )
    {
        qDebug()<<"two client connected";
        emit twoClientConnected();
        this->close();
    }
}
