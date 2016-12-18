#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include "QtCore"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "vector"
#include "iostream"
#include "qlist.h"
#include "unordered_map"
#include "memory"
using namespace std;
using namespace cv;

#define GAMMA 0.50625
#define C_PARAM 312.5
#define VOCABULARY "vocabulary"
#define SMV_CLASSIFIER_NAME "classifier1"
#define UNKNOWN_CLASS_LABEL -1
#define UNKNOWN_CLASS_NAME "unknown class"

class Classifier
{
    private:
        static const int DEFAULT_CLUSTERS_COUNT = 100;
        const QString RESULT_DIRECTORY = "/result/";
        QString trainingDirectoryPath = "D:/Study/2 semester/CV/division-images-on-classes/train";
        QString testDirectoryPath = "D:/Study/2 semester/CV/division-images-on-classes/test";
        QString dictionaryPath = "D:/Study/2 semester/CV/division-images-on-classes/dictionary.yml";
        QString classifierStorePath;
        vector<string> getAllImageNamesFromDirectory(QString directoryPath);
        vector<string> classes;
        int clustersCount;
        int determineClassLabel(const string &imageName);
    public:

        Classifier();
        void setTrainingDirectoryPath(const QString path);
        void setTestDirectoryPath(const QString path);
        void setDictionaryPath(const QString path);
        void setClassifierStorePath(const QString path);
        void addClassName(const string className);

        void buildDictionary();
        void train();
        void testDivide();
        void setClustersCount(const int clustersCount);
};

#endif // CLASSIFIER_H
