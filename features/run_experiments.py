# Written by: Anusha Balakrishnan
#Date: 11/25/14
# single_experiment('../welt-annotation-spatial.txt')
# print "KNN (k=1)"
from features.dependency import cross_validate, Classifier

print "BASELINE"
cross_validate('../welt-annotation-spatial.txt', Classifier.baseline, k=5, folds=10, small_dataset=True, range_start=10, range_end=90)
print "---------------------------- "
cross_validate('../welt-annotation-spatial.txt', Classifier.knn, k=1, folds=10, small_dataset=True, range_start=10, range_end=90)
print "---------------------------- "
print "KNN (k=5)"
cross_validate('../welt-annotation-spatial.txt', Classifier.knn, k=5, folds=10, small_dataset=True, range_start=10, range_end=90)
print "---------------------------- "
print "LINEAR SVM"
cross_validate('../welt-annotation-spatial.txt', Classifier.linear_svm, k=5, folds=10, small_dataset=True, range_start=10, range_end=90)
print "---------------------------- "
print "DECISION TREE"
cross_validate('../welt-annotation-spatial.txt', Classifier.decision_tree, k=5, folds=10, small_dataset=True, range_start=10, range_end=90)

