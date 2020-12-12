#include "controller.h"
#include "config/gameconfig.h"
#include "gui/gameview.h"
#include "mainwindow.h"
#include <iostream>

#include <QMessageBox>
#include <QtDebug>
#include <QSoundEffect>
#include <QStringList>


Controller::Controller(MainWindow* mainWindow, QObject *parent) : QObject(parent)
{
    this->data.init();
    this->mainWindow = mainWindow;

    connect(&timer, &QTimer::timeout, this, &Controller::countDown);
    connect(&timerEnemy, &QTimer::timeout, this, &Controller::countDownEnemy);
    client = new TcpClient(this);
    server = new TcpServer(this);

    isGameOver = false;
}

void Controller::initTimer()
{
    if(gameConfig.getMySide() == data.curSide)
    {
        this->secLeft = gameConfig.getStepSec();
        this->secLeftEnemy = gameConfig.getStepSec();
        timer.start(1000);
        timerEnemy.stop();
    }
    else
    {
        timer.stop();
        this->secLeftEnemy = gameConfig.getStepSec();
        this->secLeft = gameConfig.getStepSec();
        timerEnemy.start(1000);
    }
}

bool Controller::serverCreateHost(int port)
{
    server->listen(QHostAddress::Any, port);
    qDebug()<<"connect to host on port"<<port<<"ip:"<<serverIP;
    client->connectToHost(QHostAddress::LocalHost, port);
    connect(client, &TcpClient::connected, [=]{ qDebug() <<"connected"; });
    connect(client, &TcpClient::dataReceived, this, &Controller::dataReceived);
    return server->isListening();
}


void Controller::clientConnect(const QHostAddress &serverIP, int port)
{
    qDebug()<<"connect to host on port"<<port<<"ip:"<<serverIP;

    client->keepConnectingToHost(serverIP, port);

    connect(client, &TcpClient::connected, [=]{ qDebug() <<"connected"; });
    connect(client, &TcpClient::dataReceived, this, &Controller::dataReceived);
}

void Controller::dataReceived(QByteArray array, qint64 len)
{
    QDataStream in(&array, QIODevice::ReadOnly);

    int msgType;
    in >> msgType;
    MsgType type = MsgType(msgType);
    if(type == MOVE)
    {
        int srcRow, srcCol, dstRow, dstCol;
        in >> srcRow >> srcCol >> dstRow >> dstCol;
        data.mapData[dstRow][dstCol] = data.mapData[srcRow][srcCol];
        data.mapData[srcRow][srcCol].init(NEUTRAL, ITEM_NONE);
        data.curSide = gameConfig.getMySide();
        data.src = QPoint(srcRow, srcCol);
        data.dst = QPoint(dstRow, dstCol);

        secLeft = gameConfig.getStepSec();
        secLeftEnemy = gameConfig.getStepSec();
        timer.start(1000);
        timerEnemy.stop();
        mainWindow->setTimer(secLeft);

        if(isKingUnderAttack())
        {
            playMusic("AttackKing");
        }

        if(isWon())
        {
            this->win();
        }
        if(isLost())
        {
            this->lose();
        }
        updateView();
    }
    else if (msgType == GAMEOVER) {
        int winner;
        in >> winner;
        if(GameSide(winner) == gameConfig.getMySide())
        {
            this->win();
        }
        else
        {
            this->lose();
        }
    }
}
void Controller::win()
{
    playMusic("Win");
    timer.stop();
    timerEnemy.stop();
    isGameOver = true;
    mainWindow->win();
    gameView->win();

}

void Controller::lose()
{
    playMusic("Loss");
    timer.stop();
    timerEnemy.stop();
    isGameOver = true;
    mainWindow->lose();
    gameView->lose();

}
void Controller::countDown()
{
    secLeft--;
    mainWindow->setTimer(this->secLeft);
    if(secLeft <= 0)
    {
        this->surrender();
    }
}
void Controller::countDownEnemy()
{
    secLeftEnemy--;
    mainWindow->setTimerEnemy(this->secLeftEnemy);
    if(secLeftEnemy <= 0)
    {
        secLeftEnemy = 0;
    }
}

void Controller::init(GameView *gameView)
{
    data.init();
    this->gameView = gameView;
}
void Controller::loadGame(const QString &text)
{
    this->isGameOver = false;
    data = GameData::fromString(text);

    initTimer();
    updateView();
}
void Controller::surrender()
{
    QByteArray msg;
    QDataStream out(&msg, QIODevice::WriteOnly);
    int msgType = GAMEOVER;
    int winner = gameConfig.getOtherSide();

    out << qint64(0) << msgType << winner;

    out.device()->seek(0);
    out << static_cast<qint64>(msg.size()) ;

    client->write(msg);
    this->lose();
}
GameData Controller::getDisplayGameData() const
{
    return gameConfig.getMySide() == RED ? this->data : GameData::reverse(this->data);
}

void Controller::updateView()
{
    std::cout<<this->data;
    GameData displayData = getDisplayGameData();
    gameView->setView(displayData);
}
void Controller::moveItem(int srcRow, int srcCol, int dstRow, int dstCol)
{
    if(data.mapData[dstRow][dstCol].itemType != ITEM_NONE && data.mapData[dstRow][dstCol].gameSide == gameConfig.getOtherSide())
    {
        playMusic("Eat");
    }
    else if(data.mapData[dstRow][dstCol].itemType == ITEM_NONE)
    {
        playMusic("Move");
    }

    data.mapData[dstRow][dstCol] = data.mapData[srcRow][srcCol];
    data.mapData[srcRow][srcCol].init(NEUTRAL, ITEM_NONE);
    data.curSide = gameConfig.getOtherSide();
    data.src = QPoint(srcRow, srcCol);
    data.dst = QPoint(dstRow, dstCol);

    // if attacking king
    if(isAttackingKing())
    {
        playMusic("AttackKing");
    }

    this->timer.stop();
    this->timerEnemy.start(1000);

    // write on the socket
    if(client)
    {
        QByteArray msg;
        QDataStream out(&msg, QIODevice::WriteOnly);
        int msgType = MOVE;
        out << qint64(0) << msgType << srcRow <<srcCol <<dstRow <<dstCol;
        out.device()->seek(0);
        out << static_cast<qint64>(msg.size()) ;
        client->write(msg);
    }
    // update view
    updateView();
    // if win
    if(isWon())
    {
        win();
    }
    if(isLost())
    {
        lose();
    }
}
void Controller::playMusic(const QString &name)
{
    if(!gameConfig.getAudioOn()) return ;
    QSoundEffect *effect = new QSoundEffect(this);
    effect->setSource(QUrl("qrc:/audios/" + name + ".wav"));
    effect->play();
}

QList<QPoint> Controller::realToDisplay(const QList<QPoint> &pl)
{
    if(gameConfig.getMySide() == RED) return pl;
    QList<QPoint> display;
    for(auto it=pl.begin();it!=pl.end();it++)
    {
        display.append(QPoint(CHESSBOARD_ROW - 1 - it->x(), CHESSBOARD_COL - 1 - it->y()));
    }
    return display;
}
QList<QPoint> Controller::getDisplayMovingRange(int srcRow, int srcCol)
{
    QList<QPoint> range = movingRange(gameConfig.getMySide(), srcRow, srcCol);
    if(gameConfig.getMySide() == RED) return range;
    for(auto it=range.begin();it!=range.end();it++)
    {
        it->setX(CHESSBOARD_ROW - 1 - it->x());
        it->setY(CHESSBOARD_COL - 1 - it->y());
    }
    return range;
}
bool Controller::isWon()
{
    if(locateKing(gameConfig.getOtherSide()) == QPoint(-1,-1))
    {
        return true;
    }
    return false;
}
bool Controller::isLost()
{
    if(locateKing(gameConfig.getMySide()) == QPoint(-1,-1))
    {
        return true;
    }
    return false;
}
QPoint Controller::locateKing(GameSide side)
{
    for(int row=0; row<CHESSBOARD_ROW; row++)
    {
        for(int col=0; col<CHESSBOARD_COL; col++)
        {
            if(data.mapData[row][col].itemType == ITEM_KING && data.mapData[row][col].gameSide == side)
            {
                return QPoint(row,col);
            }
        }
    }
    return QPoint(-1,-1);
}

QList<QPoint> Controller::locateAllPieces(GameSide side)
{
    QList<QPoint> list;
    for(int row=0; row<CHESSBOARD_ROW; row++)
    {
        for(int col=0; col<CHESSBOARD_COL; col++)
        {
            if(data.mapData[row][col].gameSide == side && data.mapData[row][col].itemType != ITEM_NONE)
            {
                list.append(QPoint(row,col));
            }
        }
    }
    return list;
}
bool Controller::isAttackingKing()
{
    // locate enemy king
    QPoint kingCoord = locateKing(gameConfig.getOtherSide());
    QList<QPoint> allMyPiecesCoord = locateAllPieces(gameConfig.getMySide());
    for(auto it=allMyPiecesCoord.begin(); it!=allMyPiecesCoord.end(); it++)
    {
        if(movingRange(gameConfig.getMySide(), it->x(), it->y()).contains(kingCoord))
        {
            return true;
        }
    }
    return false;
}

bool Controller::isKingUnderAttack()
{
    // locate my king
    QPoint kingCoord = locateKing(gameConfig.getMySide());
    QList<QPoint> allOtherPiecesCoord = locateAllPieces(gameConfig.getOtherSide());
    for(auto it=allOtherPiecesCoord.begin(); it!=allOtherPiecesCoord.end(); it++)
    {
        if(movingRange(gameConfig.getOtherSide(), it->x(), it->y()).contains(kingCoord))
        {
            return true;
        }
    }
    return false;
}

bool Controller::isLegalMove(GameSide side, ItemType type, int srcRow, int srcCol, int dstRow, int dstCol)
{
    // filter requests out of range
    if(dstCol < 0 || dstCol >= CHESSBOARD_COL || dstRow <0 || dstRow >= CHESSBOARD_ROW)
    {
        return false;
    }
    // destination is my chess piece
    if(data.mapData[dstRow][dstCol].gameSide == side)
    {
        return false;
    }
    // process specific requests
    switch (type) {
    case ITEM_ADVISOR:
        return isLegalMoveAdvisor(side, srcRow, srcCol, dstRow, dstCol);
    case ITEM_CANNON:
        return isLegalMoveCannon(side, srcRow, srcCol, dstRow, dstCol);
    case ITEM_BISHOP:
        return isLegalMoveBishop(side, srcRow, srcCol, dstRow, dstCol);
    case ITEM_KING:
        return isLegalMoveKing(side, srcRow, srcCol, dstRow, dstCol);
    case ITEM_KNIGHT:
        return isLegalMoveKnight(side, srcRow, srcCol, dstRow, dstCol);
    case ITEM_PAWN:
        return isLegalMovePawn(side, srcRow, srcCol, dstRow, dstCol);
    case ITEM_ROOK:
        return isLegalMoveRook(side, srcRow, srcCol, dstRow, dstCol);
    default:
        return false;
    }
}
bool Controller::isLegalMoveAdvisor(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol)
{
    // delta row and delta col must be both 1
    if(qAbs(dstCol - srcCol) != 1 || qAbs(dstRow - srcRow) != 1)
    {
        return false;
    }
    if(side == RED)
    {
        if(dstRow < 7 || dstCol < 3 || dstCol > 5 )
        {
            return false;
        }
    }
    else
    {
        if(dstRow > 2 || dstCol < 3 || dstCol > 5 )
        {
            return false;
        }
    }
    return true;
}
bool Controller::isLegalMoveKing(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol)
{
    // two king on the same column
    QPoint enemyKingCoord = locateKing(GameConfig::getOppositeSide(side));
    if(dstRow == enemyKingCoord.x() && dstCol == enemyKingCoord.y() && dstCol == srcCol)
    {
        int minRow = qMin(dstRow, srcRow);
        int maxRow = qMax(dstRow, srcRow);
        // aimed at enemy king, nothing in the middle
        for(int row = minRow + 1; row < maxRow; row++)
        {
            if(data.mapData[row][dstCol].itemType != ITEM_NONE)
            {
                return false;
            }
        }
        return true;
    }

    // either delta row or delta col must be 1
    if(qAbs(dstCol - srcCol) + qAbs(dstRow - srcRow) > 1)
    {
        return false;
    }
    if(side == RED)
    {
        if(dstRow < 7 || dstCol < 3 || dstCol > 5 )
        {
            return false;
        }
    }
    else
    {
        if(dstRow > 2 || dstCol < 3 || dstCol > 5 )
        {
            return false;
        }
    }
    return true;
}
bool Controller::isLegalMoveKnight(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol)
{
    // delta row + delta col == 3
    int deltaRow = qAbs(srcRow - dstRow);
    int deltaCol = qAbs(srcCol - dstCol);
    if(deltaRow + deltaCol != 3 || deltaCol == 0 || deltaRow == 0)
    {
        return false;
    }
    // another piece stands in the way
    if(deltaCol == 2)
    {
        // _______
        // |  |  |
        // -------
        int midCol = (srcCol + dstCol) / 2;
        if(data.mapData[srcRow][midCol].itemType != ITEM_NONE)
        {
            return false;
        }
    }
    else
    {
        // ____
        // |  |
        // ----
        // |  |
        // ----
        int midRow = (srcRow + dstRow) / 2;
        if(data.mapData[midRow][srcCol].itemType != ITEM_NONE)
        {
            return false;
        }
    }
    return true;
}
bool Controller::isLegalMoveBishop(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol)
{
    if(side == RED)
    {
        if(dstRow < 5 || qAbs(dstCol - srcCol) != 2 ||  qAbs(dstRow - srcRow) != 2)
        {
            return false;
        }
    }
    else
    {
        if(dstRow > 4 || qAbs(dstCol - srcCol) != 2 ||  qAbs(dstRow - srcRow) != 2)
        {
            return false;
        }
    }
    // another piece stands in the way
    int midRow = (srcRow + dstRow) / 2;
    int midCol = (srcCol + dstCol) / 2;
    if(data.mapData[midRow][midCol].itemType != ITEM_NONE)
    {
        return false;
    }
    return true;
}

bool Controller::isLegalMoveCannon(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol)
{
    // must be moving in straight line
    if(srcRow != dstRow && srcCol != dstCol)
    {
        return false;
    }
    // the middle chess pieces should be either 0 or 2
    // if counted on itself, it'll be 1 or 3

    // horizontally moving
    if(dstRow == srcRow)
    {
        int cnt=0;
        int minCol = qMin(dstCol, srcCol);
        int maxCol = qMax(dstCol, srcCol);

        for(int col = minCol; col <= maxCol; col++)
        {
            if(data.mapData[dstRow][col].itemType != ITEM_NONE)
            {
                ++cnt;
            }
        }

        if((cnt != 1 && cnt != 3) ||
                (cnt == 3 && data.mapData[dstRow][dstCol].itemType == ITEM_NONE))
        {
            return false;
        }
    }
    // vertically moving, now dstCol == srcCol
    else
    {
        int cnt = 0;
        int minRow = qMin(dstRow, srcRow);
        int maxRow = qMax(dstRow, srcRow);

        for(int row = minRow; row <= maxRow; row++)
        {
            if(data.mapData[row][dstCol].itemType != ITEM_NONE)
            {
                ++cnt;
            }
        }
        if((cnt != 1 && cnt != 3) || (cnt == 3 && data.mapData[dstRow][dstCol].itemType == ITEM_NONE))
        {
            return false;
        }
    }
    return true;
}
bool Controller::isLegalMovePawn(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol)
{
    if(side == RED)
    {
        // not across the river yet
        if(srcRow > 4)
        {
            if(srcRow - dstRow != 1 || srcCol != dstCol)
            {
                return false;
            }
        }
        // already across the river
        else if(dstRow > srcRow || qAbs(dstRow - srcRow) + qAbs(dstCol - srcCol) > 1)
        {
            return false;
        }
    }
    else
    {
        // not across the river yet
        if(srcRow < 5)
        {

            if(dstRow - srcRow != 1 || srcCol != dstCol)
            {
                return false;
            }
        }
        // aready across the river
        else if(dstRow < srcRow || qAbs(dstRow - srcRow) + qAbs(dstCol - srcCol) > 1)
        {
            return false;
        }
    }

    return true;
}
bool Controller::isLegalMoveRook(GameSide side, int srcRow, int srcCol, int dstRow, int dstCol)
{
    if(dstRow != srcRow && dstCol != srcCol)
    {
        return false;
    }
    // horizontally moving
    if(dstRow == srcRow)
    {
        int minCol = qMin(dstCol, srcCol);
        int maxCol = qMax(dstCol, srcCol);

        for(int col = minCol + 1; col < maxCol; col++)
        {

            if(data.mapData[dstRow][col].itemType != ITEM_NONE)
            {
                return false;
            }
        }
    }
    // vertically moving, now dstCol == srcCol
    else
    {
        int minRow = qMin(dstRow, srcRow);
        int maxRow = qMax(dstRow, srcRow);

        for(int row = minRow + 1; row < maxRow; row++)
        {
            if(data.mapData[row][dstCol].itemType != ITEM_NONE)
            {
                return false;
            }
        }
    }
    return true;
}
QList<QPoint> Controller::movingRange(GameSide side, int srcRow, int srcCol)
{
    switch (data.mapData[srcRow][srcCol].itemType) {
    case ITEM_ADVISOR:
        return movingRangeAdvisor(side, srcRow, srcCol);
    case ITEM_CANNON:
        return movingRangeCannon(side, srcRow, srcCol);
    case ITEM_BISHOP:
        return movingRangeBishop(side, srcRow, srcCol);
    case ITEM_KING:
        return movingRangeKing(side, srcRow, srcCol);
    case ITEM_KNIGHT:
        return movingRangeKnight(side, srcRow, srcCol);
    case ITEM_PAWN:
        return movingRangePawn(side, srcRow, srcCol);
    case ITEM_ROOK:
        return movingRangeRook(side, srcRow, srcCol);
    default:
        return QList<QPoint>();
    }
}
QList<QPoint> Controller::movingRangeAdvisor(GameSide side, int srcRow, int srcCol)
{
    QList<QPoint> list;
    for(int i=-1; i<=1; i+=2)
    {
        for(int j=-1; j<=1; j+=2)
        {
            int dstRow = srcRow + i;
            int dstCol = srcCol + j;
            if(isLegalMove(side, ITEM_ADVISOR, srcRow, srcCol, dstRow, dstCol))
            {
                list.append(QPoint(dstRow, dstCol));
            }
        }
    }
    return list;
}
QList<QPoint> Controller::movingRangeKing(GameSide side, int srcRow, int srcCol)
{
    QList<QPoint> list;
    QVector<QPoint> relativePos({QPoint(1, 0), QPoint(-1, 0), QPoint(0, 1), QPoint(0, -1),
                                 locateKing(GameConfig::getOppositeSide(side)) - QPoint(srcRow, srcCol)});

    for(int i=0; i<relativePos.size(); i++)
    {
        if(isLegalMove(side, ITEM_KING, srcRow, srcCol, srcRow + relativePos[i].x(), srcCol + relativePos[i].y()))
            list.append(QPoint(srcRow + relativePos[i].x(), srcCol + relativePos[i].y()));
    }
    return list;
}
QList<QPoint> Controller::movingRangeKnight(GameSide side, int srcRow, int srcCol)
{
    QList<QPoint> list;
    QVector<QPoint> relativePos({QPoint(1, 2), QPoint(1, -2), QPoint(-1, 2), QPoint(-1, -2),
                                 QPoint(2, 1), QPoint(2, -1), QPoint(-2, 1), QPoint(-2, -1)});

    for(int i=0; i<relativePos.size(); i++)
    {
        if(isLegalMove(side, ITEM_KNIGHT, srcRow, srcCol, srcRow + relativePos[i].x(), srcCol + relativePos[i].y()))
            list.append(QPoint(srcRow + relativePos[i].x(), srcCol + relativePos[i].y()));
    }
    return list;
}
QList<QPoint> Controller::movingRangeBishop(GameSide side, int srcRow, int srcCol)
{
    QList<QPoint> list;
    QVector<QPoint> relativePos({QPoint(2, 2), QPoint(2, -2), QPoint(-2, 2), QPoint(-2, -2)});
    for(int i=0; i<relativePos.size(); i++)
    {
        if(isLegalMove(side, ITEM_BISHOP, srcRow, srcCol, srcRow + relativePos[i].x(), srcCol + relativePos[i].y()))
            list.append(QPoint(srcRow + relativePos[i].x(), srcCol + relativePos[i].y()));
    }
    return list;
}
QList<QPoint> Controller::movingRangeCannon(GameSide side, int srcRow, int srcCol)
{
    QList<QPoint> list;
    // horizontally moving
    for(int dstCol=0; dstCol < CHESSBOARD_COL; dstCol++)
    {
        if(isLegalMove(side, ITEM_CANNON, srcRow, srcCol, srcRow, dstCol))
        {
            list.append(QPoint(srcRow, dstCol));
        }
    }
    // vertically moving
    for(int dstRow=0; dstRow < CHESSBOARD_ROW; dstRow++)
    {
        if(isLegalMove(side, ITEM_CANNON, srcRow, srcCol, dstRow, srcCol))
        {
            list.append(QPoint(dstRow, srcCol));
        }
    }
    return list;
}
QList<QPoint> Controller::movingRangePawn(GameSide side, int srcRow, int srcCol)
{
    QList<QPoint> list;
    QVector<QPoint> relativePos({QPoint(-1, 0), QPoint(0, -1), QPoint(0, 1), QPoint(1, 0)});
    for(int i=0; i<relativePos.size(); i++)
    {
        if(isLegalMove(side, ITEM_PAWN, srcRow, srcCol, srcRow + relativePos[i].x(), srcCol + relativePos[i].y()))
            list.append(QPoint(srcRow + relativePos[i].x(), srcCol + relativePos[i].y()));
    }
    return list;
}
QList<QPoint> Controller::movingRangeRook(GameSide side, int srcRow, int srcCol)
{
    QList<QPoint> list;
    // horizontally moving
    for(int dstCol=0; dstCol < CHESSBOARD_COL; dstCol++)
    {
        if(isLegalMove(side, ITEM_ROOK, srcRow, srcCol, srcRow, dstCol))
        {
            list.append(QPoint(srcRow, dstCol));
        }
    }
    // vertically moving
    for(int dstRow=0; dstRow < CHESSBOARD_ROW; dstRow++)
    {
        if(isLegalMove(side, ITEM_ROOK, srcRow, srcCol, dstRow, srcCol))
        {
            list.append(QPoint(dstRow, srcCol));
        }
    }
    return list;
}
