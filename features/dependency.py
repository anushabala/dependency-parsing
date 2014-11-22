# Written by: Anusha Balakrishnan
# Date: 10/6/14
from collections import defaultdict
from heapq import heappush, nsmallest
import math
import operator
import pickle

from enum import Enum



columns = {"index": 0, "word": 1, "stem": 2, "morph": 3, "pos": 4, "head": 5, "dep": 6}

# Use Weka - try different classification algs for the dependency labels, for parser action
# start using differeent method when you have more training data?
#
# todo: handle cases where LA or RA is predicted but can't occur - count as misclassification?
# todo: write model and training data separately
# todo baseline: labels: use most common label; arc: attach every word to previous word (left branching)

from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from features.data_tools import DataParser, get_property
from features.feature_extraction import FeatureExtractor
from features.nivre import Parser



actions = {"S": 1, "LA": 2, "RA": 3, "END": 4}
training_fvs = []

class Classifier(Enum):
    baseline = 0
    knn = 1
    linear_svm = 2
    rbf_svm = 3
    decision_tree = 4


class ParseClassifier:
    def __init__(self, k=1, mode= Classifier.knn):

        self.extractor = FeatureExtractor()
        self.training_data = []
        self.action_classifier = None
        self.dep_classifier = None
        self.k = k
        self.type = mode
        self.baseline_dep = "det"

    def get_training_data(self):
        return self.training_data

    def train(self, data):
        self.add_fv_mappings(data)
        training_fvs = self.extractor.convert_to_fvs(data)
        self.training_data = training_fvs
        if self.type== Classifier.linear_svm:
            self.__train_svm('linear')
            # self.write_training_data(filepath, training_fvs, model=trained_svm)
        elif self.type== Classifier.rbf_svm:
            self.__train_svm('rbf')
        elif self.type == Classifier.decision_tree:
            self.__train_decision_tree()
        elif self.type == Classifier.knn:
            self.__train_knn()
        elif self.type != Classifier.baseline:
            print "Selected classifier type unknown. Please choose from one of the options in the Classifier enum."
            exit(1)

    def add_fv_mappings(self, data):
        for state in data:
            self.extractor.add_fv_mappings(state)

    def write_fv_mappings(self, filepath):
        self.extractor.write_fv_mappings(filepath)


    def write_model(self, filepath, dep_classifier_path=None):
        if self.type=="knn":
            pickle.dump(self.training_data, filepath)
        elif self.type=="svm":
            pickle.dump(self.action_classifier, filepath)
            pickle.dump(self.dep_classifier, dep_classifier_path)
        elif self.type=="linear_svm":
            pickle.dump(self.action_classifier, filepath)
            pickle.dump(self.dep_classifier, dep_classifier_path)

    def __train_svm(self, kernel='linear'):
        self.action_classifier = SVC(kernel=kernel)
        self.dep_classifier = SVC(kernel=kernel)
        action_labels = [f[0] for f in self.training_data]
        dep_labels = [f[1] for f in self.training_data]
        fvs = [f[2].values() for f in self.training_data]
        new_fvs = []
        for f in fvs:
            f = self.extractor.convert_to_values(f)
            new_fvs.append(f)
        fvs = new_fvs

        self.action_classifier.fit(fvs, action_labels)
        self.dep_classifier.fit(fvs, dep_labels)

    def load_model(self, filepath, dep_classifier_path=None):
        if self.type=="knn":
            self.training_data = pickle.load(filepath)
        elif self.type=="linear_svm":
            self.action_classifier = pickle.load(filepath)
            self.dep_classifier = pickle.load(dep_classifier_path)
        elif self.type=="decision_tree":
            self.action_classifier = pickle.load(filepath)
            self.dep_classifier = pickle.load(dep_classifier_path)

    def get_next_action(self, state):

        fv_state = self.extractor.convert_instance_to_fv(state)
        if self.type == Classifier.linear_svm or self.type == Classifier.rbf_svm:
            (chosen_action, chosen_dep) = self.__svm(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])
        elif self.type == Classifier.decision_tree:
            (chosen_action, chosen_dep) = self.__decision_tree(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])
        elif self.type == Classifier.knn:
            (chosen_action, chosen_dep) = self.__sklearn_knn(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])
        elif self.type == Classifier.baseline:
            (chosen_action, chosen_dep) = self.__baseline(fv_state)

    def __train_decision_tree(self):
        self.action_classifier = tree.DecisionTreeClassifier()
        self.dep_classifier = tree.DecisionTreeClassifier()
        action_labels = [f[0] for f in self.training_data]
        dep_labels = [f[1] for f in self.training_data]
        fvs = [f[2].values() for f in self.training_data]
        new_fvs = []
        for f in fvs:
            f = self.extractor.convert_to_values(f)
            new_fvs.append(f)
        fvs = new_fvs

        self.action_classifier.fit(fvs, action_labels)
        self.dep_classifier.fit(fvs, dep_labels)
    def __train_knn(self):
        self.action_classifier = KNeighborsClassifier(n_neighbors=self.k)
        self.dep_classifier = KNeighborsClassifier(n_neighbors=self.k)
        action_labels = [f[0] for f in self.training_data]
        dep_labels = [f[1] for f in self.training_data]
        fvs = [f[2].values() for f in self.training_data]
        new_fvs = []
        for f in fvs:
            f = self.extractor.convert_to_values(f)
            new_fvs.append(f)
        fvs = new_fvs

        self.action_classifier.fit(fvs, action_labels)
        self.dep_classifier.fit(fvs, dep_labels)
    def ___knn(self, fv_state):
        min_distances = []
        for j in range(0, len(self.training_data)):
            f = self.training_data[j]
            train_vectors = f[2]
            train_dist = 0.0
            for i in range(0, len(train_vectors)):
                train_v = train_vectors[i]
                test_v = fv_state[i]
                dist = 0.0
                for c in range(0, len(train_v)):
                    diff = math.fabs(int(train_v[c]) - int(test_v[c])) ** 2
                    dist += diff
                dist = math.sqrt(dist)
                train_dist += dist
            heappush(min_distances, (train_dist, j))
        smallest = nsmallest(self.k, min_distances)
        next_action = defaultdict(int)

        next_dep = defaultdict(int)
        for (dist, pos) in smallest:
            fv = self.training_data[pos]
            action_num = fv[0]
            dep_num = fv[1]
            next_action[action_num] += 1
            next_dep[dep_num] += 1

        chosen_action = max(next_action.iteritems(), key=operator.itemgetter(1))[0]
        chosen_dep = max(next_dep.iteritems(), key=operator.itemgetter(1))[0]

        if chosen_action == actions["S"] and chosen_dep != 0:
            chosen_dep = 0
        elif chosen_action != actions["S"] and chosen_dep == 0:
            chosen_dep = \
                max([(key, value) for (key, value) in next_dep.iteritems() if value != 0], key=operator.itemgetter(1))[0]

        return (chosen_action, chosen_dep)

    def __svm(self, fv_state):
        state_values = self.extractor.convert_to_values(fv_state.values())
        pred_action = self.action_classifier.predict(state_values)[0]
        pred_dep = self.dep_classifier.predict(state_values)[0]

        return (pred_action, pred_dep)

    def __decision_tree(self, fv_state):
        state_values = self.extractor.convert_to_values(fv_state.values())
        pred_action = self.action_classifier.predict(state_values)[0]
        pred_dep = self.dep_classifier.predict(state_values)[0]
        return (pred_action, pred_dep)

    def __sklearn_knn(self, fv_state):
        state_values = self.extractor.convert_to_values(fv_state.values())
        pred_action = self.action_classifier.predict(state_values)[0]
        pred_dep = self.dep_classifier.predict(state_values)[0]
        return (pred_action, pred_dep)


def train(filepath, train_file, print_status=False, model=Classifier.knn):
    # print("[Training]")
    global FV_MAPPINGS
    FV_MAPPINGS = defaultdict(lambda: ["NULL"])
    training_data = []
    parser = Parser()
    classifier = ParseClassifier(mode=model)
    infile = open(filepath, 'r')
    line = infile.readline()
    first = True
    properties = {}
    sentence = None
    num = 0

    while line:
        line = line.strip()
        line = line.lower()
        if line == "":
            num +=1
            parser.reset()
            sent_features = parser.get_state_sequence(sentence, properties)
            training_data.append(sent_features)
            if print_status:
                print "%d:\t%s" % (num, sentence)
            properties = {}
            first = True
            break
        elif first:
            line = line.split()
            line = [w.strip() for w in line]
            sentence = line
            first = False
        else:
            line = line.split('\t')
            line = [w.strip() for w in line]
            pos = int(line[columns["index"]])
            properties[pos] = line[columns["index"] + 1:]
        line = infile.readline()
    infile.close()

    for sent_features in training_data:
        for f in sent_features:
            print f

    classifier.train(training_data)
    training_fvs = classifier.get_training_data()
    writer = DataParser()
    writer.write_fvs(training_fvs, train_file)

    return classifier
    # print "Completed training"


def get_raw_accuracy(predictions, test_dependencies):
    num = 0
    total = 0.0
    correct_dep = 0.0
    correct_arc = 0.0

    for key in test_dependencies.keys():
        for (head, label, dep) in test_dependencies[key]:
            total += 1
            for (pred_head, pred_label, pred_dep) in predictions[key]:
                if head == pred_head and dep == pred_dep:
                    correct_arc += 1
                    if label == pred_label:
                        correct_dep += 1

    dep_accuracy = correct_dep / total
    arc_accuracy = correct_arc/total
    return (dep_accuracy, arc_accuracy)

def get_f1(predictions, test_dependencies):
    pass

def get_precision(predictions, test_dependencies):
    true_arc = 0.0
    labeled_arc = 0.0
    true_dep = 0.0
    labeled_dep = 0.0

    # for key in test_dependencies.keys():
    #     for (head, label, dep) in test_dependencies[key]:
    #         total += 1
    #         for (pred_head, pred_label, pred_dep) in predictions[key]:
    #             if head == pred_head and dep == pred_dep:
    #                 correct_arc += 1
    #                 if label == pred_label:
    #                     correct_dep += 1
    #
    # dep_accuracy = correct_dep / total
    # arc_accuracy = correct_arc/total
    # return (dep_accuracy, arc_accuracy)

def get_recall(predictions, test_dependencies):
    pass

def get_dependencies_from_properties(real_properties):
    test_dependencies = {}
    for key in real_properties.keys():
        dependencies = real_properties[key]
        all_deps = []
        for index in dependencies:
            head = get_property(dependencies, index, "head")
            label = get_property(dependencies, index, "dep")
            all_deps.append((head, label, index))
        test_dependencies[key] = all_deps
    return test_dependencies

def predict(filepath, model_path=None, print_status=False, k=1, classifier=None, mode=Classifier.knn):
    parser = Parser()
    if classifier==None:
        classifier = ParseClassifier(k, mode=mode)
        classifier.load_model(model_path)

    infile = open(filepath, 'r')
    line = infile.readline()
    first = True
    properties = defaultdict(list)
    test_properties = defaultdict(list)
    tested_sentences = []
    sentence = None
    num = 0
    real_properties = {}
    predictions = {}
    while line:
        line = line.strip()
        line = line.lower()
        if line == "":

            parser.reset()
            num += 1
            tested_sentences.append(" ".join(sentence))
            sent_pred = parser.predict_actions(sentence, properties, classifier)
            predictions[num - 1] = sent_pred
            real_properties[num - 1] = test_properties
            if print_status and num%5==0:
                print "%d:\t%s" % (num, sentence)

            properties = defaultdict(list)
            test_properties = defaultdict(list)
            first = True

        elif first:

            line = line.split()
            line = [w.strip() for w in line]
            sentence = line
            first = False
        else:
            line = line.split('\t')
            line = [w.strip() for w in line]
            pos = int(line[columns["index"]])
            properties[pos] = line[columns["index"] + 1:]
            properties[pos][-1] = 'NULL'
            properties[pos][-2] = 'NULL'

            test_properties[pos] = line[columns["index"] + 1:]

        line = infile.readline()

    infile.close()
    real_dependencies = get_dependencies_from_properties(real_properties)
    return (predictions, real_dependencies)

def incremental_train(filepath, mode, k=1, folds=5):
    parser = DataParser()
    num_data = parser.load_data(filepath)
    print num_data
    train_file = '../all_training_data.txt'
    test_file = '../test_data.txt'
    parser.initial_split(train_file, test_file)
    for i in range(10, 91, 10):
        train_num = int((i/100.0) * num_data)
        # print train_num
        (dep_accuracy, arc_accuracy) = cross_validate(train_file, train_num, mode, test_file, k, folds)
        print "Number of training instances: %d (%d%%)%%\t%2.3f\t%2.3f" \
              % (train_num, i, dep_accuracy, arc_accuracy)

def cross_validate(filepath, num_train, mode, test_file, k=1, folds=5):
    parser = DataParser()
    parser.load_data(filepath)
    train_file = '../training_data.txt'
    train_fvs = '../training.dat'
    dep_total = 0.0
    arc_total = 0.0
    for j in range(0, folds):
        (num_train, num_test) = parser.choose_training_data(train_file, num_train)
        # print "\t", num_train, num_test
        if k<0:
            k = max(int(num_train/2) - 1, 1)
        classifier = train(train_file, train_fvs, model=mode)
        (predictions, real_dependencies) = predict(test_file, k=k, classifier=classifier, mode=mode)
        (dep_accuracy, arc_accuracy) = get_raw_accuracy(predictions, real_dependencies)

        dep_total += dep_accuracy
        arc_total += arc_accuracy

    return (dep_total/folds, arc_total/folds)


def single_experiment(filepath):
    parser = DataParser()
    parser.load_data(filepath)
    train_file = '../all_training_data.txt'
    test_file = '../test_data.txt'
    parser.initial_split(train_file, test_file)
    classifier = train(train_file, '../training.dat', model=Classifier.decision_tree)
    # (predictions, real_dependencies) = predict('../welt-annotation-spatial.txt', '../training.dat', classifier=classifier)
    #
    # accuracy = get_raw_accuracy(predictions, real_dependencies)
    # print accuracy

single_experiment('../welt-annotation-spatial.txt')
# predict('../welt-annotation-spatial.txt', start=11, max=1 print_status=True, k=4)
# print "KNN (k=1)"
# incremental_train('../welt-annotation-spatial.txt', Classifier.knn, k=1, folds=10)
# print "---------------------------- "
# print "KNN (k=5)"
# incremental_train('../welt-annotation-spatial.txt', Classifier.knn, k=5, folds=10)
# print "---------------------------- "
# print "KNN (k= n/2)"
# incremental_train('../welt-annotation-spatial.txt', Classifier.knn, k=-1, folds=10)
# print "---------------------------- "
# print "LINEAR SVM"
# incremental_train('../welt-annotation-spatial.txt', Classifier.linear_svm, k=-1, folds=10)
# print "---------------------------- "
# print "RBF SVM"
# incremental_train('../welt-annotation-spatial.txt', Classifier.rbf_svm, k=-1, folds=10)
# print "---------------------------- "
# print "DECISION TREE"
# incremental_train('../welt-annotation-spatial.txt', Classifier.decision_tree, k=-1, folds=10)
