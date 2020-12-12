#ifndef GAMEVIEW_H
#define GAMEVIEW_H


#include <QGraphicsView>
#include "gui/itemchess.h"
#include "gui/gamescene.h"
#include "core/controller.h"

class GameView : public QGraphicsView
{
public:
    GameView(Controller* controller, QWidget *parent = nullptr);

    void mousePressEvent(QMouseEvent *event) override;

    void setView(const GameData &data);
    void win();
    void lose();


private:
    GameScene scene;
    QVector<QVector<ItemChess*>> chessPieces;
    QList<QPoint> realRangeList;
    QList<QPoint> displayRangeList;
    ItemChess *pieceChosen;
    ItemChess *pieceOldPlace;
    Controller *controller;

};

#endif // GAMEVIEW_H
