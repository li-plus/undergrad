#include "mainwindow.h"
#include <QApplication>
#include "darkstyle/darkstyle.h"

#include <time.h>


int main(int argc, char *argv[])
{
    srand((unsigned)time(nullptr));

    QApplication a(argc, argv);
    a.setStyle(new DarkStyle);
    MainWindow w;
    w.show();

    return a.exec();
}
