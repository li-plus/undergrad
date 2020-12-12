#include "gamedata.h"
#include "config/gameconfig.h"
#include <QtDebug>
#include <QPoint>

GameData::GameData()
{
    mapData = QVector<QVector<BlockData>> (10, QVector<BlockData>(9));
    curSide = RED;
    src = QPoint(-1,-1);
    dst = QPoint(-1,-1);
    for(int row=0;row<CHESSBOARD_ROW;row++)
    {
        for(int col=0;col<CHESSBOARD_COL;col++)
        {
            mapData[row][col].setCoord(row, col);
        }
    }
}

void GameData::initEmpty()
{
    mapData = QVector<QVector<BlockData>> (10, QVector<BlockData>(9));
    curSide = RED;
    src = QPoint(-1,-1);
    dst = QPoint(-1,-1);
    for(int row=0;row<CHESSBOARD_ROW;row++)
    {
        for(int col=0;col<CHESSBOARD_COL;col++)
        {
            mapData[row][col].setCoord(row, col);
        }
    }
}
GameData GameData::reverse(const GameData &data)
{
    GameData reversedData(data);
    for(int row=0;row<CHESSBOARD_ROW; row ++)
    {
        std::reverse(reversedData.mapData[row].begin(), reversedData.mapData[row].end());
    }
    std::reverse(reversedData.mapData.begin(), reversedData.mapData.end());
    return reversedData;
}

QString GameData::toString() const
{
    QString dataString;
    QTextStream out(&dataString);

    QVector<QList<QPoint>> piecePointList (14, QList<QPoint>());

    for(int row = 0; row < CHESSBOARD_ROW; row++)
    {
        for(int col = 0; col < CHESSBOARD_COL; col++)
        {
            if(mapData[row][col].gameSide == curSide)
            {
                piecePointList[mapData[row][col].itemType].append(
                            QPoint(col, CHESSBOARD_ROW - 1 - row));
            }
            else if(mapData[row][col].gameSide == GameConfig::getOppositeSide(curSide))
            {
                piecePointList[7 + mapData[row][col].itemType].append(
                            QPoint(col, CHESSBOARD_ROW - 1 - row));
            }
        }
    }

    out << (curSide == RED ? "red\n" : "black\n");
    for(int i=0;i<14;i++)
    {
        if(i == 7)
        {
            out<<(curSide == BLACK ? "red\n" : "black\n");
        }
        out << QString::number(piecePointList[i].size());
        for(auto it = piecePointList[i].begin(); it!=piecePointList[i].end(); it++)
        {
            out<< " <" << it->x() << ',' << it->y() << '>';
        }
        out<<"\n";
    }

    return dataString;
}
GameData GameData::fromString(const QString &text)
{
    GameData data;
    data.initEmpty();
    QStringList infoList = text.split("\n", QString::SkipEmptyParts);
    GameSide side;
    side = infoList.at(0).toLower() == QString("red") ? RED : BLACK;

    data.curSide = side;

    for(int i=0;i<7;i++)
    {
        if(infoList.at(i+1).at(0) == "0") continue;
        QStringList coordList(infoList.at(i+1).split(" ", QString::SkipEmptyParts));
        for(auto it = coordList.begin() + 1; it != coordList.end(); it++)
        {
            QStringList coord = it->split(QRegExp("[<,>]"), QString::SkipEmptyParts);
            qDebug()<<coord[0].toInt()<<coord[1].toInt();
            int row = CHESSBOARD_ROW - 1 - coord[1].toInt();
            int col = coord[0].toInt();
            data.mapData[row][col].init(side, ItemType(ITEM_KING + i));
        }
    }

    side = infoList.at(8).toLower() == QString("red") ? RED : BLACK;
    for(int i=0;i<7;i++)
    {
        if(infoList.at(i+1+8).at(0) == "0") continue;
        QStringList coordList(infoList.at(i+1+8).split(" ", QString::SkipEmptyParts));
        for(auto it = coordList.begin() + 1; it != coordList.end(); it++)
        {
            QStringList coord = it->split(QRegExp("[<,>]"), QString::SkipEmptyParts);
            int row = CHESSBOARD_ROW - 1 - coord[1].toInt();
            int col = coord[0].toInt();
            data.mapData[row][col].init(side, ItemType(ITEM_KING + i));
        }
    }
    return data;
}
void GameData::init()
{
    // init red
    mapData[9][0].init(RED, ITEM_ROOK);
    mapData[9][1].init(RED, ITEM_KNIGHT);
    mapData[9][2].init(RED, ITEM_BISHOP);
    mapData[9][3].init(RED, ITEM_ADVISOR);
    mapData[9][4].init(RED, ITEM_KING);
    mapData[9][5].init(RED, ITEM_ADVISOR);
    mapData[9][6].init(RED, ITEM_BISHOP);
    mapData[9][7].init(RED, ITEM_KNIGHT);
    mapData[9][8].init(RED, ITEM_ROOK);

    mapData[6][0].init(RED, ITEM_PAWN);
    mapData[6][2].init(RED, ITEM_PAWN);
    mapData[6][4].init(RED, ITEM_PAWN);
    mapData[6][6].init(RED, ITEM_PAWN);
    mapData[6][8].init(RED, ITEM_PAWN);

    mapData[7][1].init(RED, ITEM_CANNON);
    mapData[7][7].init(RED, ITEM_CANNON);


    // init black

    mapData[0][0].init(BLACK, ITEM_ROOK);
    mapData[0][1].init(BLACK, ITEM_KNIGHT);
    mapData[0][2].init(BLACK, ITEM_BISHOP);
    mapData[0][3].init(BLACK, ITEM_ADVISOR);
    mapData[0][4].init(BLACK, ITEM_KING);
    mapData[0][5].init(BLACK, ITEM_ADVISOR);
    mapData[0][6].init(BLACK, ITEM_BISHOP);
    mapData[0][7].init(BLACK, ITEM_KNIGHT);
    mapData[0][8].init(BLACK, ITEM_ROOK);

    mapData[3][0].init(BLACK, ITEM_PAWN);
    mapData[3][2].init(BLACK, ITEM_PAWN);
    mapData[3][4].init(BLACK, ITEM_PAWN);
    mapData[3][6].init(BLACK, ITEM_PAWN);
    mapData[3][8].init(BLACK, ITEM_PAWN);

    mapData[2][1].init(BLACK, ITEM_CANNON);
    mapData[2][7].init(BLACK, ITEM_CANNON);
}
