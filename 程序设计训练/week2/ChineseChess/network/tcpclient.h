#ifndef TCPCLIENT_H
#define TCPCLIENT_H

#include <QObject>
#include <QTcpSocket>
#include <QTimer>
//    TODO
//    signals needed to be connected
//    TcpClient::connected()
//    TcpClient::disconnected(int)
//    TcpClient::dataReceived(QByteArray, qint64)
//
//    FUNCTIONS
//    connectToHost()
//    write()
//    disconnectFromHost()
//    localAddress()


class TcpClient : public QTcpSocket
{
    Q_OBJECT
public:
    TcpClient(QObject *parent = nullptr);

signals:
    void dataReceived(QByteArray, qint64);
    void disconnected(int);

public slots:
    void slotReadyRead();
    void slotDisconnected();
    void keepConnectingToHost(const QHostAddress &address, quint16 port, OpenMode mode = ReadWrite);
    void cancelConnectingToHost();
    void sendSue();
public:
    inline qint64 write(const QByteArray &data){ return QIODevice::write(data); }

private:
    QTimer timer;
};

#endif // TCPCLIENT_H
