#include "damage_detection.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    damage_detection w;
    w.show();
    return a.exec();
}
