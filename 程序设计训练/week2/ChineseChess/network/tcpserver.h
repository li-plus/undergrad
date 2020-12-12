#ifndef TCPSERVER_H
#define TCPSERVER_H

#include <QObject>
#include <QTcpServer>
#include <QTcpSocket>
// listen to given port


//    TODO:
//    signals needed to be connected
//    void updated(QString, int);
//
//    FUNCTIONS
//    serverAddress()

class TcpServer : public QTcpServer
{
    Q_OBJECT
public:
    TcpServer(QObject *parent = nullptr);
    TcpServer(int port, QObject *parent = nullptr);
    // save connections with every client
    QList<QTcpSocket*> tcpClientList;

signals:
    void updated(QByteArray, int);
    void twoClientConnected();

public slots:
    void reply(qintptr handle, QByteArray, int);
    void updateServer(QByteArray, int);
    void slotDisconnected(int);
    void receiveBytes(QTcpSocket *tcpClient);
protected:
    virtual void incomingConnection(qintptr handle) override;
};

#endif // TCPSERVER_H
