#include "gamescene.h"

#include <QPainter>


GameScene::GameScene(QObject *parent) : QGraphicsScene (parent)
{

}
void GameScene::drawBackground(QPainter *painter, const QRectF &rect)
{
    painter->save();
    QPixmap pixmapBGD(":/images/WOOD.GIF");
    painter->drawPixmap(this->sceneRect(), pixmapBGD, pixmapBGD.rect());

    painter->restore();
}
