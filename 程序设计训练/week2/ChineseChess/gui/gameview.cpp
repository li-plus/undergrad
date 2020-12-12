#include "gameview.h"

#include <QMouseEvent>
#include <QtDebug>
#include <QMessageBox>


#include "mainwindow.h"

#include "config/gameconfig.h"

#include "definition.h"

GameView::GameView(Controller* ctrl, QWidget *parent) :
    QGraphicsView (parent)
{
    // init var
    this->pieceChosen = nullptr;
    this->controller = ctrl;
    this->chessPieces = QVector<QVector<ItemChess*>>(CHESSBOARD_ROW, QVector<ItemChess*>(CHESSBOARD_COL));
    this->scene.setSceneRect(-30,-30,540,600);
    this->setScene(&scene);
    this->setFixedSize(550,610);

    QPainter painter(this);
    QPixmap chessBoard(":/images/WOOD.GIF");
    painter.drawPixmap(scene.sceneRect(), chessBoard, chessBoard.rect());

    GameData data = controller->getDisplayGameData();

    // this is the display row and col
    for(int row=0;row<CHESSBOARD_ROW;row++)
    {
        for(int col=0;col<CHESSBOARD_COL;col++)
        {
            chessPieces[row][col] = new ItemChess(data.mapData[row][col].gameSide, data.mapData[row][col].itemType,
                                                  row, col, data.mapData[row][col].row, data.mapData[row][col].col);
            scene.addItem(chessPieces[row][col]);
        }
    }
}

void GameView::mousePressEvent(QMouseEvent *event)
{
    if(controller->getIsGameOver() || controller->getDisplayGameData().curSide != gameConfig.getMySide())
    {
        qDebug()<<"forbid mouse event";
        return ;
    }
    QPointF pos = this->mapToScene(event->pos());
    // device transform is the transformation between scene and view
    if(event->button() == Qt::LeftButton)
    {
        if(QGraphicsItem *item = scene.itemAt(pos, QTransform()))
        {
            if(ItemChess* itemChess = dynamic_cast<ItemChess*>(item))
            {
                // select piece from my side
                if(itemChess->getSide() == gameConfig.getMySide())
                {
                    controller->playMusic("Choose");
                    // already chose a piece
                    if(pieceChosen)
                    {
                        pieceChosen->unselect();
                        for(auto it=displayRangeList.begin(); it!=displayRangeList.end(); it++)
                        {
                            chessPieces[it->x()][it->y()]->setHint(false);
                        }
                    }
                    itemChess->select();
                    this->pieceChosen = itemChess;
                    this->realRangeList = controller->movingRange(gameConfig.getMySide(), pieceChosen->getRealRow(), pieceChosen->getRealCol());
                    this->displayRangeList = controller->realToDisplay(this->realRangeList);
                    if(gameConfig.getShowHint())
                    {
                        for(auto it= displayRangeList.begin(); it!=displayRangeList.end(); it++)
                        {
                            chessPieces[it->x()][it->y()]->setHint(true);
                        }
                    }
                }
                // select space piece or other side's piece
                else
                {
                    // moving or eating
                    if(pieceChosen)
                    {
                        // legal move
                        if(realRangeList.contains(QPoint(itemChess->getRealRow(), itemChess->getRealCol())))
                        {
                            controller->moveItem(pieceChosen->getRealRow(), pieceChosen->getRealCol(),
                                                 itemChess->getRealRow(), itemChess->getRealCol());
                            this->pieceChosen = nullptr;
                        }
                        // illegal move
                        else
                        {
                            controller->playMusic("Forbid");
                        }
                    }
                }
            }
        }
    }
    else if(event->button() == Qt::RightButton)
    {
        if(pieceChosen) pieceChosen->unselect();
        pieceChosen = nullptr;
        for(auto it= displayRangeList.begin(); it!=displayRangeList.end(); it++)
        {
            chessPieces[it->x()][it->y()]->setHint(false);
        }
    }
    scene.update();
}

void GameView::win()
{
    QMessageBox::information(this, tr("Info"), tr("You Win!"));
}
void GameView::lose()
{
    QMessageBox::information(this, tr("Info"), tr("You Lose!"));
}
void GameView::setView(const GameData &data)
{
    for(int row=0; row<CHESSBOARD_ROW; row++)
    {
        for(int col=0; col<CHESSBOARD_COL; col++)
        {
            chessPieces[row][col]->init(data.mapData[row][col].gameSide, data.mapData[row][col].itemType, row, col,
                                        chessPieces[row][col]->getRealRow(), chessPieces[row][col]->getRealCol(), false, false, false);
        }
    }
    if(data.src.x() >= 0 && data.dst.x() >= 0)
    {
        if(gameConfig.getMySide() == RED)
        {
            chessPieces[data.src.x()][data.src.y()]->setTrace(true);
            chessPieces[data.dst.x()][data.dst.y()]->setTrace(true);
        }
        else
        {
            chessPieces[CHESSBOARD_ROW - 1 - data.src.x()][CHESSBOARD_COL - 1 - data.src.y()]->setTrace(true);
            chessPieces[CHESSBOARD_ROW - 1 - data.dst.x()][CHESSBOARD_COL - 1 - data.dst.y()]->setTrace(true);
        }
    }

    scene.update();
}
