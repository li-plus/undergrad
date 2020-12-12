#include "itemchess.h"

#include <QPainter>
#include <QtDebug>

ItemChess::ItemChess(GameSide side, ItemType type, int displayRow, int displayCol, int realRow, int realCol)
{
    // init var
    this->type = type;
    this->displayRow = displayRow;
    this->displayCol = displayCol;
    this->realCol = realCol;
    this->realRow = realRow;
    this->side = side;
    this->isSelected = false;
    this->isHidden = false;
    this->isHint = false;
    this->isTrace = false;

    // init pixmap
    setPixmap(false);

    // set pos
    this->setPos(CHESSPIECE_SIZE * displayCol, CHESSPIECE_SIZE * displayRow);
}
void ItemChess::init(GameSide side, ItemType type, int displayRow, int displayCol, int realRow, int realCol,
                     bool isSelected, bool isHidden, bool isTrace)
{
    // init var
    this->type = type;
    this->displayRow = displayRow;
    this->displayCol = displayCol;
    this->realCol = realCol;
    this->realRow = realRow;
    this->side = side;
    this->isHint = false;
    this->isHidden = isHidden;
    this->isTrace = isTrace;
    if(isSelected)
    {
        this->select();
    }
    else
    {
        this->unselect();
    }

    // init pixmap
    setPixmap(this->isSelected);

    // set pos
    this->setPos(CHESSPIECE_SIZE * displayCol, CHESSPIECE_SIZE * displayRow);
}

QPixmap ItemChess::getHighlightPixmap()
{
    return QPixmap(":/images/WOOD/HL.png");
}

QPixmap ItemChess::getFramePixmap()
{
    return QPixmap(":/images/WOOD/OOS.GIF");
}
void ItemChess::setPixmap(bool selected)
{
    QString dirWood = ":/images/WOOD/";
    QString sideCode = side2code[side];
    QString typeCode = type2code[type];
    if(selected) typeCode += "S";
    pixmapItem = QPixmap(dirWood + sideCode + typeCode + ".GIF");
}
QPixmap ItemChess::getPixmap(bool selected)
{
    QString dirWood = ":/images/WOOD/";
    QString sideCode = side2code[side];
    QString typeCode = type2code[type];
    if(selected) typeCode += "S";
    return QPixmap(dirWood + sideCode + typeCode + ".GIF");
}
void ItemChess::unselect()
{
    // already unselected
    if(!this->isSelected) return;

    this->isSelected = false;
    setPixmap(false);
    killTimer(timerID);
    this->isHidden = false;
}
void ItemChess::select()
{
    // already selected
    if(this->isSelected) return;

    this->isSelected = true;
    setPixmap(true);
    timerID = startTimer(500);
}

void ItemChess::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing);
    if(isTrace)
    {
        QPixmap pixmapSelected = getPixmap(true);
        painter->drawPixmap(boundingRect(), pixmapSelected, pixmapSelected.rect());
    }
    if(this->isHidden)
    {
        QPixmap pixmapFrame = ItemChess::getFramePixmap();
        painter->drawPixmap(boundingRect(), pixmapFrame, pixmapFrame.rect());
    }
    else
    {
        painter->drawPixmap(boundingRect(), pixmapItem, pixmapItem.rect());
        if(this->isHint)
        {
            QPixmap pixmapHint = ItemChess::getHighlightPixmap();
            painter->drawPixmap(boundingRect(), pixmapHint, pixmapHint.rect());
        }
    }
    painter->restore();
}
void ItemChess::timerEvent(QTimerEvent *event)
{
    this->isHidden = !this->isHidden;
    update();
}


QRectF ItemChess::boundingRect() const
{
    return QRectF(-CHESSPIECE_SIZE * 0.5, -CHESSPIECE_SIZE * 0.5, CHESSPIECE_SIZE, CHESSPIECE_SIZE);
}
