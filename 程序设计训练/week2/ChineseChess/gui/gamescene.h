#ifndef GAMESCENE_H
#define GAMESCENE_H

#include <QGraphicsScene>


class GameScene : public QGraphicsScene
{
    Q_OBJECT
public:
    GameScene(QObject *parent = nullptr);
    virtual void drawBackground(QPainter *painter, const QRectF &rect);
};

#endif // GAMESCENE_H
