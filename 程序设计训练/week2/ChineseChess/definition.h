#ifndef DEFINITION_H
#define DEFINITION_H


#include <QMap>
#include <QString>

enum ItemType{ ITEM_KING = 0, ITEM_ADVISOR, ITEM_BISHOP, ITEM_KNIGHT, ITEM_ROOK, ITEM_CANNON, ITEM_PAWN, ITEM_NONE};

enum GameSide{ RED, BLACK, NEUTRAL };

enum AppType { SERVER, CLIENT };

enum MsgType { MOVE, GAMEOVER, SUE };

static QMap<ItemType, QString> type2code =
{
    {ITEM_BISHOP, "B"}, {ITEM_CANNON, "C"}, {ITEM_ROOK, "R"}, {ITEM_KING, "K"},
    {ITEM_ADVISOR, "A"}, {ITEM_KNIGHT, "N"}, {ITEM_PAWN, "P"}, {ITEM_NONE, "O"}
};
static QMap<GameSide, QString> side2code =
{
    {RED, "R"}, {BLACK, "B"}, {NEUTRAL, "O"}
};

const int CHESSBOARD_ROW = 10;
const int CHESSBOARD_COL = 9;
static const int CHESSPIECE_SIZE = 60;
static const int CHESSBOARD_EDGE = 4;



#endif // DEFINITION_H
