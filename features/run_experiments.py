# Written by: Anusha Balakrishnan
#Date: 11/25/14
# single_experiment('../welt-annotation-spatial.txt')
# print "KNN (k=1)"
from features.dependency import Classifier, ExperimentRunner

def german():
    corpus_file = '../treebanks/german_train.conll'
    # runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    # print "GERMAN - BASELINE"
    # runner.cross_validate(corpus_file, Classifier.baseline, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)

    print "---------------------------- "
    runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    print "GERMAN - KNN (k=1)"
    runner.cross_validate(corpus_file, Classifier.knn, k=1, folds=1, small_dataset=False, range_start=8, range_end=8, exponential=True)

    # print "---------------------------- "
    # runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    # print "GERMAN - LINEAR SVM"
    # runner.cross_validate(corpus_file, Classifier.linear_svm, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)

    # print "---------------------------- "
    # runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    # print "GERMAN - DECISION TREE"
    # runner.cross_validate(corpus_file, Classifier.decision_tree, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)

def slovene():
    corpus_file = '../treebanks/slovene_train.conll'
    runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    print "SLOVENE - BASELINE"
    runner.cross_validate(corpus_file, Classifier.baseline, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)

    # print "---------------------------- "
    # runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    # print "SLOVENE - KNN (k=1)"
    # runner.cross_validate(corpus_file, Classifier.knn, k=1, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)

    print "---------------------------- "
    runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    print "SLOVENE - LINEAR SVM"
    runner.cross_validate(corpus_file, Classifier.linear_svm, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)

    print "---------------------------- "
    runner = ExperimentRunner(lang="german", conll=True, fine_tags=False)
    print "SLOVENE - DECISION TREE"
    runner.cross_validate(corpus_file, Classifier.decision_tree, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)


def portugese():
    corpus_file = "../treebanks/portugese_train.conll"
    # print "---------------------------- "
    # runner = ExperimentRunner(lang="portugese", conll=True, fine_tags=False)
    # print "PORTUGESE - BASELINE"
    # runner.cross_validate(corpus_file, Classifier.baseline, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)

    # print "--------------------------- "
    # runner = ExperimentRunner(lang="portugese", conll=True, fine_tags=False)
    # print "PORTUGESE - KNN"
    # runner.cross_validate(corpus_file, Classifier.knn, k=1, folds=10, small_dataset=False, range_start=9, range_end=9, exponential=True)

    print "---------------------------- "
    runner = ExperimentRunner(lang="portugese", conll=True, fine_tags=False)
    print "PORTUGESE - LINEAR SVM"
    runner.cross_validate(corpus_file, Classifier.linear_svm, k=5, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)

    print "---------------------------- "
    runner = ExperimentRunner(lang="portugese", conll=True, fine_tags=False)
    print "PORTUGESE - DECISION TREE"
    runner.cross_validate(corpus_file, Classifier.decision_tree, k=5, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)

def dutch():
    corpus_file = "../treebanks/dutch_train.conll"
    print "---------------------------- "
    runner = ExperimentRunner(lang="dutch", conll=True, fine_tags=False)
    print "DUTCH - BASELINE"
    runner.cross_validate(corpus_file, Classifier.baseline, k=5, folds=10, small_dataset=False, range_start=0, range_end=10, exponential=True)


    # print "---------------------------- "
    # runner = ExperimentRunner(lang="dutch", conll=True, fine_tags=False)
    # print "DUTCH - KNN (k=1)"
    # runner.cross_validate(corpus_file, Classifier.knn, k=1, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)
    #
    # print "---------------------------- "
    # runner = ExperimentRunner(lang="dutch", conll=True, fine_tags=False)
    # print "DUTCH - LINEAR SVM"
    # runner.cross_validate(corpus_file, Classifier.linear_svm, k=1, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)
    #
    # print "---------------------------- "
    # runner = ExperimentRunner(lang="dutch", conll=True, fine_tags=False)
    # print "DUTCH - DECISION TREE"
    # runner.cross_validate(corpus_file, Classifier.decision_tree, k=5, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)

def danish():
    corpus_file = "../treebanks/danish_train.conll"

    # print "---------------------------- "
    # runner = ExperimentRunner(lang="danish", conll=True, fine_tags=False)
    # print "DANISH - BASELINE"
    # runner.cross_validate(corpus_file, Classifier.baseline, k=5, folds=10, small_dataset=False, range_start=1, range_end=10, exponential=True)
    #
    # print "---------------------------- "
    # runner = ExperimentRunner(lang="danish", conll=True, fine_tags=False)
    # print "DANISH - KNN (k=1)"
    # runner.cross_validate(corpus_file, Classifier.knn, k=1, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)
    #
    # print "---------------------------- "
    # runner = ExperimentRunner(lang="danish", conll=True, fine_tags=False)
    # print "DANISH - LINEAR SVM"
    # runner.cross_validate(corpus_file, Classifier.linear_svm, k=5, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)
    #
    print "---------------------------- "
    runner = ExperimentRunner(lang="danish", conll=True, fine_tags=False)
    print "DANISH - DECISION TREE"
    runner.cross_validate(corpus_file, Classifier.decision_tree, k=5, folds=10, small_dataset=False, range_start=1, range_end=10, exponential=True)

def swedish():
    corpus_file = "../treebanks/swedish_train.conll"

    # print "---------------------------- "
    # runner = ExperimentRunner(lang="swedish", conll=True, fine_tags=False)
    # print "SWEDISH - BASELINE"
    # runner.cross_validate(corpus_file, Classifier.baseline, k=5, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)

    # print "---------------------------- "
    # runner = ExperimentRunner(lang="swedish", conll=True, fine_tags=False)
    # print "SWEDISH - KNN"
    # runner.cross_validate(corpus_file, Classifier.knn, k=1, folds=10, small_dataset=False, range_start=10, range_end=10, exponential=True)

    print "---------------------------- "
    runner = ExperimentRunner(lang="swedish", conll=True, fine_tags=False)
    print "SWEDISH - LINEAR SVM"
    runner.cross_validate(corpus_file, Classifier.linear_svm, k=5, folds=10, small_dataset=False, range_start=9, range_end=10, exponential=True)

    # print "---------------------------- "
    # runner = ExperimentRunner(lang="swedish", conll=True, fine_tags=False)
    # print "SWEDISH - DECISION TREE"
    # runner.cross_validate(corpus_file, Classifier.decision_tree, k=5, folds=10, small_dataset=False, range_start=0, range_end=0, exponential=True)

def annotated_data():
    corpus_file = '../welt-data/welt-train.txt'
    # print "---------------------------- "
    # runner = ExperimentRunner("en", conll=False, fine_tags=False)
    # print "ANNOTATED DATA - BASELINE"
    # runner.cross_validate(corpus_file, Classifier.baseline, k=5, folds=10, small_dataset=False, range_start=0, range_end=7, exponential=True)

    print "---------------------------- "
    runner = ExperimentRunner(lang="en", conll=False, fine_tags=False)
    print "ANNOTATED DATA - KNN"
    runner.cross_validate(corpus_file, Classifier.knn, k=1, folds=10, small_dataset=False, range_start=0, range_end=7, exponential=True)
    #
    # print "---------------------------- "
    # runner = ExperimentRunner(lang="en", conll=False, fine_tags=False)
    # print "ANNOTATED DATA - LINEAR SVM"
    # runner.cross_validate(corpus_file, Classifier.linear_svm, k=5, folds=10, small_dataset=False, range_start=0, range_end=7, exponential=True)
    # #
    # print "---------------------------- "
    # runner = ExperimentRunner(lang="en", conll=False, fine_tags=False)
    # print "ANNOTATED DATA - DECISION TREE"
    # runner.cross_validate(corpus_file, Classifier.decision_tree, k=5, folds=10, small_dataset=False, range_start=0, range_end=7, exponential=True)



# annotated_data()

# danish()

# dutch()

german()

# portugese()

# swedish()
#
# slovene()