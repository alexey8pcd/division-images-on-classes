#include <QCoreApplication>
#include "classifier.h"
#include "iostream"

using namespace std;

int main(int argc, char *argv[]) {
    cout << "Start working" << endl;
    QString classifierStorePath
            = "D:/Study/2 semester/CV/division-images-on-classes/classifier.xml";
    Classifier classifier = Classifier(classifierStorePath);
    QString trainPath = QString("D:/Study/2 semester/CV/division-images-on-classes/cvTrainDir");
//    QString trainPath = QString("D:/Study/2 semester/CV/division-images-on-classes/cvTrainDir/ext");
    classifier.setTrainingDirectoryPath(trainPath);
    classifier.setDictionaryPath("D:/Study/2 semester/CV/division-images-on-classes/dictionary.yml");
    QString testPath = QString("D:/Study/2 semester/CV/division-images-on-classes/cvTestDir");
    classifier.setTestDirectoryPath(testPath);
    classifier.buildDictionary();
    classifier.train();
    classifier.testDivide();
    cout << "Working finished";
    return 0;
}

