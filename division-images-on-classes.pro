QT += core
QT -= gui

TARGET = division-images-on-classes
CONFIG += console
CONFIG -= app_bundle
QMAKE_CXXFLAGS += -std=c++14
TEMPLATE = app

SOURCES += main.cpp \
    classifier.cpp

HEADERS += \
    classifier.h \
    classes.h

INCLUDEPATH += D:/Storage/Programs/Libraries/opencv/build/install/include
LIBS += D:/Storage/Programs/Libraries/opencv/build/bin/*.dll
