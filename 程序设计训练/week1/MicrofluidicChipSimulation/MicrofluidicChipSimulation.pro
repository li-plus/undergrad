#-------------------------------------------------
#
# Project created by QtCreator 2018-08-27T19:20:38
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MicrofluidicChipSimulation
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

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    gui/pipe.cpp \
    gui/paintarea.cpp \
    core/coresimulator.cpp \
    gui/pipeioconfigdlg.cpp \
    gui/pipenode.cpp \
    darkstyle/darkstyle.cpp \
    gui/flowdesigndlg.cpp \
    core/workerevolve.cpp \
    core/simulatorapi.cpp

HEADERS += \
        mainwindow.h \
    definition.h \
    gui/pipe.h \
    gui/paintarea.h \
    core/coresimulator.h \
    core/simulatorapi.h \
    gui/pipeioconfigdlg.h \
    gui/pipenode.h \
    darkstyle/darkstyle.h \
    gui/flowdesigndlg.h \
    core/workerevolve.h

RESOURCES   += \
               darkstyle/darkstyle.qrc \

FORMS += \
        mainwindow.ui \
    gui/pipeioconfigdlg.ui \
    gui/flowdesigndlg.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
