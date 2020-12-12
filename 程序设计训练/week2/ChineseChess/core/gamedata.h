#ifndef GAMEDATA_H
#define GAMEDATA_H

#include <QVector>
#include <QDataStream>
#include <QPoint>

#include "definition.h"

#include "config/gameconfig.h"

class BlockData
{
public:
    BlockData() {
        itemType = ITEM_NONE;
        gameSide = NEUTRAL;
    }
    void init(GameSide side, ItemType type)
    {
        this->itemType = type;
        this->gameSide = side;
    }
    friend QDataStream &operator>>(QDataStream &in, BlockData &data)
    {
        int type, side;
        in >> data.row >> data.col >> type >> side;
        data.itemType = ItemType(type);
        data.gameSide = GameSide(side);
        return in;
    }
    friend QDataStream &operator<<(QDataStream &out, const BlockData &data)
    {
        int type = data.itemType;
        int side = data.gameSide;

        out<< data.row << data.col << type << side;
        return out;
    }

    void setCoord(int row, int col){ this->row = row; this->col = col; }
    int row;
    int col;
    ItemType itemType;
    GameSide gameSide;
};

class GameData
{
public:
    GameData();
    void init();
    void initEmpty();



    static GameData reverse(const GameData &data);
    static GameData fromString(const QString &text);
    QString toString() const ;

    friend std::ostream &operator<<(std::ostream &os, const GameData &data)
    {
        for(int i=0;i<CHESSBOARD_ROW;i++)
        {
            for(int j=0;j<CHESSBOARD_COL;j++)
            {
                if(data.mapData[i][j].itemType == ITEM_NONE) os<<"  ";
                else os<<data.mapData[i][j].itemType<<' ';
            }
            os<<std::endl;
        }
        return os;
    }

    friend QDataStream &operator>>(QDataStream &in, GameData &data)
    {
        int side;
        in >> data.mapData >> side;
        data.curSide = GameSide(side);
        return in;
    }

    friend QDataStream &operator<<(QDataStream &out, const GameData &data)
    {
        int side = data.curSide;
        out << data.mapData << side;
        return out;
    }


    QVector<QVector<BlockData>> mapData;
    GameSide curSide;
    QPoint src, dst;
};

#endif // GAMEDATA_H
