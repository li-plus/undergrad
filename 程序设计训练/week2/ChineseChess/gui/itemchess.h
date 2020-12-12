#ifndef ITEMCHESS_H
#define ITEMCHESS_H


#include <QGraphicsItem>
#include <QPixmap>
#include <QTimer>


#include "definition.h"


class ItemChess : public QObject, public QGraphicsItem
{

public:
    ItemChess(GameSide side, ItemType type, int row, int col, int realRow, int realCol);

    static QPixmap getFramePixmap();
    static QPixmap getHighlightPixmap();

    virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

    virtual QRectF boundingRect() const override;
    virtual void timerEvent(QTimerEvent *event) override;

    void select();
    void unselect();

    // writer
    void setPixmap(bool selected);
    void init(GameSide side, ItemType type, int row, int col, int realRow, int realCol, bool isSelected, bool isHidden, bool isTrace);
    void setHint(bool hint) { this->isHint = hint; }
    void setTrace(bool trace) { this->isTrace = trace; }
    QPixmap getPixmap(bool selected);
    // reader
    GameSide getSide() const { return side; }
    ItemType getItemType() const { return type; }
    bool getIsSelected() const {return isSelected;}
    int getDisplayRow() const { return displayRow; }
    int getDisplayCol() const { return displayCol; }
    int getRealRow() const { return realRow; }
    int getRealCol() const { return realCol; }
    int getIsTrace() const { return isTrace; }
private:
    // independent var
    GameSide side;
    ItemType type;
    int displayRow, displayCol;
    int realRow, realCol;
    // dependent var
    QPixmap pixmapItem;
    bool isTrace;
    bool isSelected;
    bool isHidden;
    bool isHint;
    int timerID;
};

#endif // ITEMCHESS_H
