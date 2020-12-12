#ifndef PIPESCENE_H
#define PIPESCENE_H

#include <QObject>
#include <QGraphicsScene>
class PipeScene : public QGraphicsScene
{
    Q_OBJECT
public:
    PipeScene(QObject *parent = nullptr);
};

#endif // PIPESCENE_H
