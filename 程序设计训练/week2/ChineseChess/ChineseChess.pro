#-------------------------------------------------
#
# Project created by QtCreator 2018-09-04T09:46:46
#
#-------------------------------------------------

QT       += core gui widgets multimedia network

TARGET = ChineseChess
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    gui/gameview.cpp \
    gui/itemchess.cpp \
    core/controller.cpp \
    gui/gamescene.cpp \
    config/gameconfig.cpp \
    core/gamedata.cpp \
    gui/settingsdlg.cpp \
    network/tcpclient.cpp \
    network/tcpserver.cpp \
    gui/clientconnectdlg.cpp \
    gui/servercreatehostdlg.cpp

HEADERS += \
        mainwindow.h \
    gui/gameview.h \
    gui/itemchess.h \
    core/controller.h \
    definition.h \
    gui/gamescene.h \
    config/gameconfig.h \
    core/gamedata.h \
    gui/settingsdlg.h \
    network/tcpclient.h \
    network/tcpserver.h \
    gui/clientconnectdlg.h \
    gui/servercreatehostdlg.h

FORMS += \
        mainwindow.ui \
    gui/settingsdlg.ui \
    gui/clientconnectdlg.ui \
    gui/servercreatehostdlg.ui

RESOURCES += \
    qtchess.qrc

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
