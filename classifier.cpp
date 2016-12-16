#include "classifier.h"
#include "classes.h"

Classifier::Classifier(QString classifierStorePath) {
    this->clustersCount = DEFAULT_CLUSTERS_COUNT;
    this->classifierStorePath = classifierStorePath;
}

vector<string> Classifier::getAllImageNamesFromDirectory(QString directoryPath){
    QStringList filters;
    filters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp";
    QDir dir(directoryPath);
    QFileInfoList fileInfoList = dir.entryInfoList(
                                     filters, QDir::Files|QDir::NoDotAndDotDot);
    vector<string> imageNames;
    QListIterator<QFileInfo> it(fileInfoList);
    while (it.hasNext()) {
        const QFileInfo info = it.next();
        const QString filePath = info.canonicalFilePath();
        const string stdStringFilePath = filePath.toStdString();
        imageNames.push_back(stdStringFilePath);
    }
    return imageNames;
}

void Classifier::setTrainingDirectoryPath(const QString path) {
    this->trainingDirectoryPath = path;
}

void Classifier::setTestDirectoryPath(const QString path) {
    this->testDirectoryPath = path;
}

void Classifier::setDictionaryPath(const QString path){
    this->dictionaryPath = path;
}

void Classifier::setClassifierStorePath(const QString path) {
    this->classifierStorePath = path;
}

void Classifier::buildDictionary(){
    cout << "Start building dictionary" << endl;
    vector<KeyPoint> keypointsStore;
    Mat features;
    Mat descriptorsStore;
    SurfDescriptorExtractor detector;
    vector<string> imageNames = getAllImageNamesFromDirectory(trainingDirectoryPath);
    for (string imageName: imageNames) {
        const Mat image = imread(imageName);
        detector.detect(image, keypointsStore);
        detector.compute(image, keypointsStore, descriptorsStore);
        features.push_back(descriptorsStore);
        cout << "Extract information from image: " << imageName << endl;
    }
    cout << "Clustering..." << endl;
    double epsilon = 0.001;
    int maxCount = 100;
    TermCriteria termCriteria(CV_TERMCRIT_ITER, maxCount, epsilon);
    int retries = 1;
    BOWKMeansTrainer bowTrainer(clustersCount, termCriteria, retries, KMEANS_PP_CENTERS);
    Mat dictionary = bowTrainer.cluster(features);
    FileStorage fileStorage(dictionaryPath.toStdString(), FileStorage::WRITE);
    fileStorage << VOCABULARY << dictionary;
    fileStorage.release();
    cout << "Dictionary built"  << endl;
}

void Classifier::train() {
    cout << "Prepare to training" << endl;
    Mat dictionary;
    FileStorage fileStorage(dictionaryPath.toStdString(), FileStorage::READ);
    fileStorage[VOCABULARY] >> dictionary;
    fileStorage.release();

    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher());
    Ptr<DescriptorExtractor> dextractor(new SurfDescriptorExtractor());
    BOWImgDescriptorExtractor bowDescriptorExtractor(dextractor, matcher);
    bowDescriptorExtractor.setVocabulary(dictionary);
    SurfFeatureDetector surfDetector;
    vector<string> imageNames = getAllImageNamesFromDirectory(trainingDirectoryPath);
    Mat trainingData(0, imageNames.size(), CV_32SC1);
    Mat labels(0, 1, CV_32SC1);
    cout << "Make training data" << endl;
    int count = 0, total = imageNames.size();
    for (string imageName: imageNames) {
        const Mat image = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        vector<KeyPoint> keyPoints;
        surfDetector.detect(image, keyPoints);
        Mat bowDescriptor;
        bowDescriptorExtractor.compute(image, keyPoints, bowDescriptor);
        trainingData.push_back(bowDescriptor);
        if (imageName.find(CLASS_EGRET_PREFIX) != string::npos){
            labels.push_back(CLASS_EGRET_LABEL);
        } else if (imageName.find(CLASS_OWL_PREFIX) != string::npos) {
            labels.push_back(CLASS_OWL_LABEL);
        } else if (imageName.find(CLASS_TOUCAN_PREFIX) != string::npos) {
            labels.push_back(CLASS_TOUCAN_LABEL);
        } else {
            labels.push_back(CLASS_UNKNOWN_LABEL);
        }
        ++count;
        cout << count << "/" << total << endl;
    }
    CvSVMParams params;
    params.kernel_type = CvSVM::RBF;
    params.svm_type = CvSVM::C_SVC;
    params.gamma = GAMMA;
    params.C = C_PARAM;
    double epsilon = 0.000001;
    int maxCount = 100;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, maxCount, epsilon);
    CvSVM svmClassifier;
    cout << "Start training" << endl;
    bool result = svmClassifier.train(trainingData, labels, Mat(), Mat(), params);
    if (result) {
        cout << "Training success, save classifier" << endl;
        const char * fileName = classifierStorePath.toLatin1().constData();
        svmClassifier.save(fileName, SMV_CLASSIFIER_NAME);
        cout << "Classifier " << SMV_CLASSIFIER_NAME
             << " saved from " << fileName << endl;
    } else {
        cout << "Get error while training" << endl;
    }
}

void Classifier::testDivide() {
    cout << "Test divide images" << endl;
    Mat dictionary;
    FileStorage fileStorage(dictionaryPath.toStdString(), FileStorage::READ);
    fileStorage[VOCABULARY] >> dictionary;
    fileStorage.release();

    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher());
    Ptr<DescriptorExtractor> dextractor(new SurfDescriptorExtractor());
    BOWImgDescriptorExtractor bowDescriptorExtractor(dextractor, matcher);
    bowDescriptorExtractor.setVocabulary(dictionary);
    SurfFeatureDetector surfDetector;
    CvSVM svmClassifier;
    const char * fileName = classifierStorePath.toLatin1().constData();
    svmClassifier.load(fileName, SMV_CLASSIFIER_NAME);
    cout << "Classifier " << SMV_CLASSIFIER_NAME
         << " loaded from " << fileName << endl;
    vector<string> testImagesNames = getAllImageNamesFromDirectory(testDirectoryPath);
    vector<KeyPoint> keyPoints2;
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    for (string imageName: testImagesNames) {
        const Mat image = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        surfDetector.detect(image, keyPoints2);
        Mat bowDescriptor;
        bowDescriptorExtractor.compute(image, keyPoints2, bowDescriptor);
        int classLabel = svmClassifier.predict(bowDescriptor);
        string className = CLASS_UNKNOWN;
        switch(classLabel){
            case CLASS_EGRET_LABEL:
                className = CLASS_EGRET_PREFIX;
                break;
            case CLASS_OWL_LABEL:
                className = CLASS_OWL_PREFIX;
                break;
            case CLASS_TOUCAN_LABEL:
                className = CLASS_TOUCAN_PREFIX;
                break;
        }
        int index = imageName.find_last_of('//');
        string simpleName = imageName;
        if(index != string::npos){
            simpleName = imageName.substr(index + 1);
        }
        string resultPath = testDirectoryPath.toStdString()+
                                RESULT_DIRECTORY.toStdString()
                            + className + "/" + simpleName;
        bool writed = imwrite(resultPath, image, compression_params);
        if (writed){
            cout << "Write image " << resultPath << endl;
        } else {
            cout << "Image predicted class " << className
                 << " to image " << imageName << endl;
        }
    }
    cout << "Test done" << endl;
}

void Classifier::setClustersCount(const int clustersCount){
    this->clustersCount = clustersCount;
}

