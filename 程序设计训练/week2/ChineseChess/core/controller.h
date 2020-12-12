#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>
#include <QTimer>
#include "core/gamedata.h"
#include "network/tcpclient.h"
#include "network/tcpserver.h"

class MainWindow;
class GameView;

class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(MainWindow* mainWindow, QObject *parent = nullptr);
    void init(GameView *gameView);
    void initTimer();
    // validation functions

    QList<QPoint> movingRange(GameSide side, int srcRow, int srcCol);

    bool isAttackingKing();
    bool isKingUnderAttack();
    bool isWon();
    bool isLost();
    void win();
    void lose();
    void playMusic(const QString &name);
    void surrender();

    // op function
    QPoint locateKing(GameSide side);
    QList<QPoint> locateAllPieces(GameSide side);
    QList<QPoint> realToDisplay(const QList<QPoint> &pl);
    QList<QPoint> getDisplayMovingRange(int srcRow, int srcCol);

    void moveItem(int srcRow, int srcCol, int dstRow, int dstCol);
    void updateView();
    void loadGame(const QString &text);

    // reader
    GameData getDisplayGameData() const;
    QString getDataString() const { return data.toString(); }
    TcpClient *getClient() const { return client; }
    TcpServer *getServer() const { return server; }
    bool getIsGameOver() const { return isGameOver; }

    // writer
    void setGameView(GameView *gameView) { this->gameView = gameView; }
    void dataReceived(QByteArray array, qint64 len);

    void countDown();
    void countDownEnemy();
    // network
    void clientConnect(const QHostAddress &serverIP, int port);
    bool serverCreateHost(int port);

private:
    bool isLegalMove(GameSide side, ItemType type, int srcRow, int srcCol, int dstRow, int dstCol);

    bool isLegalMoveAdvisor(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol);
    bool isLegalMoveKing(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol);
    bool isLegalMoveKnight(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol);
    bool isLegalMoveBishop(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol);
    bool isLegalMoveCannon(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol);
    bool isLegalMovePawn(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol);
    bool isLegalMoveRook(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol);

    QList<QPoint> movingRangeAdvisor(GameSide side, int srcRow, int srcCol);
    QList<QPoint> movingRangeKing(GameSide side, int srcRow, int srcCol);
    QList<QPoint> movingRangeKnight(GameSide side, int srcRow, int srcCol);
    QList<QPoint> movingRangeBishop(GameSide side, int srcRow, int srcCol);
    QList<QPoint> movingRangeCannon(GameSide side, int srcRow, int srcCol);
    QList<QPoint> movingRangePawn(GameSide side, int srcRow, int srcCol);
    QList<QPoint> movingRangeRook(GameSide side, int srcRow, int srcCol);


private:
    MainWindow *mainWindow = nullptr;
    GameData data;
    GameView *gameView = nullptr;
    QTimer timer, timerEnemy;
    int secLeft, secLeftEnemy;
    TcpClient *client = nullptr;
    TcpServer *server = nullptr;
    QHostAddress serverIP;
    bool isGameOver;
};

#endif // CONTROLLER_H
