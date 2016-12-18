#include <QCoreApplication>
#include "classifier.h"
#include "iostream"

using namespace std;



int main(int argc, char *argv[]) {
    QString WORK_DIR_PARAM = "workdir";
    QString TEST_DIR_PARAM = "testdir";
    QString TRAIN_DIR_PARAM = "traindir";


    cout << "Start working" << endl;

    Classifier classifier = Classifier();
    QString workPath = nullptr;
    QString trainDir = nullptr;
    QString testDir = nullptr;
    for (int i = 1; i < argc; ++i) {
        char * param = argv[i];
        QString paramQ = QString(param);
        if(paramQ.contains('=')){
            QStringList list = paramQ.split("=");
            if(list.size() != 2){
                cout << "Invalid param " << param;
                return -1;
            } else {
                QString paramName = list.at(0);
                QString paramValue = list.at(1);
                if (WORK_DIR_PARAM.compare(paramName) == 0){
                    workPath = paramValue;
                } else if (TEST_DIR_PARAM.compare(paramName) == 0){
                    testDir = paramValue;
                } else if (TRAIN_DIR_PARAM.compare(paramName) == 0){
                    trainDir = paramValue;
                }
            }
        } else {
            classifier.addClassName(paramQ.toStdString());
        }
    }
    if(workPath == nullptr || workPath.isEmpty()){
        cout << "Parameter workdir is missing";
        return -1;
    }
    if(trainDir == nullptr || trainDir.isEmpty()){
        cout << "Parameter traindir is missing";
        return -1;
    }
    if(testDir == nullptr || testDir.isEmpty()){
        cout << "Parameter testdir is missing";
        return -1;
    }
    QString classifierStorePath
            = workPath + QDir::separator() + "classifier.xml";
    QString dictionaryPath = workPath + QDir::separator() + "dictionary.yml";
    classifier.setClassifierStorePath(classifierStorePath);
    classifier.setTrainingDirectoryPath(trainDir);
    classifier.setDictionaryPath(dictionaryPath);
    classifier.setTestDirectoryPath(testDir);
    classifier.buildDictionary();
    classifier.train();
    classifier.testDivide();
    cout << "Working finished";
    return 0;
}

