# Written by: Anusha Balakrishnan
# Date: 10/6/14
from collections import defaultdict
import copy
from heapq import heappush, nsmallest
import math
import operator
import pickle

from enum import Enum
import itertools


columns = {"index": 0, "word": 1, "stem": 2, "morph": 3, "pos": 4, "head": 5, "dep": 6}

# Use Weka - try different classification algs for the dependency labels, for parser action
# start using differeent method when you have more training data?
#
# todo: handle cases where LA or RA is predicted but can't occur - count as misclassification?
# todo: write model and training data separately
# todo baseline: labels: use most common label; arc: attach every word to previous word (left branching)

from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from features.data_tools import DataParser, get_property, DataExtractor
from features.feature_extraction import FeatureExtractor
from features.nivre import Parser
from features.nivre import ParserActions
import numpy as np



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
        self.model_fvs = []
        self.action_classifier = None
        self.dep_classifier = None
        self.k = k
        self.type = mode
        self.baseline_dep = "det"
        self.scaler = None
        # todo: extra
        # self.extra_classifier = None

    def get_training_data(self):
        return self.model_fvs

    def train(self, data):
        self.add_fv_mappings(data)
        training_fvs = self.extractor.convert_to_fvs(data)
        # print self.extractor.FV_MAPPINGS
        # for fv in training_fvs[:10]:
        #     print fv
        self.training_data = training_fvs
        if self.type == Classifier.baseline:
            self.__train_baseline()
        else:
            if self.type== Classifier.linear_svm:
                self.__train_svm()
                # self.write_training_data(filepath, training_fvs, model=trained_svm)
            elif self.type == Classifier.decision_tree:
                self.__train_decision_tree()
            elif self.type == Classifier.knn:
                self.__train_knn()
            else:
                print "Selected classifier type unknown. Please choose from one of the options in the Classifier enum."
                exit(1)
            action_labels = [f[0] for f in self.training_data]
            dep_labels = [f[1] for f in self.training_data]
            fvs = [f[2].values() for f in self.training_data]
            new_fvs = []
            for f in fvs:
                f = self.extractor.convert_to_values(f)
                new_fvs.append(f)
            self.scaler = OneHotEncoder()
            self.scaler.fit(new_fvs)
            fvs_scaled = self.scaler.transform(new_fvs).toarray()
            self.action_classifier.fit(fvs_scaled, action_labels)
            self.dep_classifier.fit(fvs_scaled, dep_labels)
            #todo: extra
            # self.extra_classifier.fit(fvs_scaled, dep_labels)



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

    def __train_baseline(self):
        dep_labels = [f[1] for f in self.training_data]
        # print dep_labels
        counts = defaultdict(int)
        for label in dep_labels:
            if label!=0:
                counts[label] += 1
        self.baseline_dep = max(counts.iteritems(), key=operator.itemgetter(1))[0]
        # print self.baseline_dep


    def __train_svm(self):
        self.action_classifier = LinearSVC()
        # self.action_classifier = KNeighborsClassifier(n_neighbors=self.k, algorithm='auto')
        self.dep_classifier = LinearSVC()
        # self.model_fvs = [(a, d, f) for a,d,f in itertools.izip(action_labels, dep_labels, fvs_scaled.toarray().tolist())]


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


        if self.type == Classifier.baseline:
            (chosen_action, chosen_dep) = self.__baseline(fv_state)
        else:
            state_values = self.extractor.convert_to_values(fv_state.values())
            binarized = self.scaler.transform([state_values]).toarray()[0]
            chosen_action = self.action_classifier.predict(binarized)[0]
            chosen_dep = self.dep_classifier.predict(binarized)[0]
        # if self.type == Classifier.linear_svm or self.type == Classifier.rbf_svm:
        #     (chosen_action, chosen_dep) = self.__svm(fv_state)
        # elif self.type == Classifier.decision_tree:
        #     (chosen_action, chosen_dep) = self.__decision_tree(fv_state)
        # elif self.type == Classifier.baseline:
        #     (chosen_action, chosen_dep) = self.__baseline(fv_state)
        # else:
        #     (chosen_action, chosen_dep) = self.__sklearn_knn(fv_state)

        return (ParserActions(chosen_action), self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])

    #todo: extra
    def get_extra_dep(self, state):
        fv_state = self.extractor.convert_instance_to_fv(state)
        state_values = self.extractor.convert_to_values(fv_state.values())
        binarized = self.scaler.transform([state_values]).toarray()[0]
        #todo: extra
        # chosen_dep = self.extra_classifier.predict(binarized)[0]
        return self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep]

    def __train_decision_tree(self):
        self.action_classifier = tree.DecisionTreeClassifier()
        self.dep_classifier = tree.DecisionTreeClassifier()
    def __train_knn(self):
        self.action_classifier = KNeighborsClassifier(n_neighbors=self.k, algorithm='auto')
        self.dep_classifier = KNeighborsClassifier(n_neighbors=self.k, algorithm='auto')
        #todo: extra
        # self.extra_classifier = LinearSVC()

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
        binarized = self.scaler.transform([state_values])[0]
        pred_action = self.action_classifier.predict(binarized)[0]
        pred_dep = self.dep_classifier.predict(binarized)[0]

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

    def __baseline(self, fv_state):
        # chosen_action = ParserActions.LA.value
        chosen_action = ParserActions.RA.value
        chosen_dep = self.baseline_dep
        return (chosen_action, chosen_dep)


class ExperimentRunner:
    def __init__(self, lang, conll=True, fine_tags=False):
        self.lang = lang
        self.conll_format = conll
        self.fine_tags = fine_tags
        self.extractor = DataExtractor(self.lang, self.conll_format, self.fine_tags)

    def train(self, raw_data, morph_features, fv_file, print_status=False, model=Classifier.knn, k=1):
        # print("[Training]")
        training_data = []
        parser = Parser(morph_features)
        classifier = ParseClassifier(mode=model, k=k)
        for (sentence, properties) in raw_data:
            real_deps = self.get_dependencies_from_properties(properties)
            # print sentence
            # print real_deps
            sent_features = parser.get_state_sequence(sentence, properties, gold_standard=real_deps)
            training_data.append(sent_features)
            # for f in sent_features:
            #     print f[0], f[1]
            # dependencies = parser.A
            # print dependencies


            # print self.get_raw_accuracy({0:dependencies}, real_deps)
            # mismatches = self.get_mismatches(dependencies, real_deps)
            # if mismatches<=1:



        # print "Training instances: %d" % len(training_data)
        classifier.train(training_data)
        # training_fvs = classifier.get_training_data()
        # writer = DataParser()
        # writer.write_fvs(training_fvs, fv_file)
        #
        return classifier, morph_features

    def get_raw_accuracy(self, predictions, test_dependencies):
        num = 0
        total = 0.0
        correct_dep = 0.0
        correct_arc = 0.0

        for key in test_dependencies.keys():
            for (head, label, dep) in test_dependencies[key]:
                if label == 'punc' or label == 'punct' or label == 'pnct':
                    continue
                total += 1
                found = False
                for (pred_head, pred_label, pred_dep) in predictions[key]:
                    if head == pred_head and dep == pred_dep:
                        correct_arc += 1
                        if label == pred_label:
                            correct_dep += 1
                            found = True
                # if not found:
                #     print (head, label, dep)


        dep_accuracy = correct_dep / total
        arc_accuracy = correct_arc/total
        return (dep_accuracy, arc_accuracy)

    def get_mismatches(self, predictions, test_dependencies):
        mismatches = 0

        for (head, label, dep) in test_dependencies:
            found = False
            for (pred_head, pred_label, pred_dep) in predictions:
                if head == pred_head and dep == pred_dep:
                    if label == pred_label:
                        found = True
            if not found:
                mismatches += 1

        return mismatches

    def get_real_dependencies(self, real_properties):
        test_dependencies = {}
        for key in real_properties.keys():
            dependencies = real_properties[key]
            all_deps = self.get_dependencies_from_properties(dependencies)
            test_dependencies[key] = all_deps
        return test_dependencies

    def get_dependencies_from_properties(self, real_properties):
        all_deps = []
        for index in real_properties:
            head = get_property(real_properties, index, "head")
            label = get_property(real_properties, index, "dep")
            all_deps.append((head, label, index))
        return all_deps

    def predict(self, test_data, morph_features, classifier, print_status=False):
        parser = Parser(morph_features=morph_features)
        num = 0
        real_properties = {}
        predictions = {}
        real_dependencies = {}
        for (sentence, actual_properties) in test_data:
            # print sentence
            # print actual_properties
            num += 1
            test_properties = {}
            for key in actual_properties:
                test_properties[key] = copy.deepcopy(actual_properties[key])
                test_properties[key][-1] = 'NULL'
                test_properties[key][-2] = 'NULL'
            sent_pred = parser.predict_actions(sentence, test_properties, classifier)
            parser.reset()
            predictions[num-1] = sent_pred
            real_properties[num-1] = actual_properties
            # parser.get_state_sequence(sentence, actual_properties)
            # real_dependencies[num-1] = parser.A


        # print "All properties ", real_properties
        real_dependencies = self.get_real_dependencies(real_properties)
        # print real_dependencies
        # print predictions
        return (predictions, real_dependencies)

    def cross_validate(self, filepath, mode, k=1, folds=10, range_start=10, range_end=90, incremental=True, small_dataset=False, exponential=False):
        parser = DataParser()
        total_data = parser.load_data(filepath, limit=1200, max_sentence_size=30)
        # print "Total data: %d" % total_data
        dep_total = defaultdict(float)
        arc_total = defaultdict(float)
        for j in range(0, folds):
            print "Fold %d" % (j+1)
            data_file = '../all_training_data_%s.txt' % self.lang
            test_file = '../test_data_%s.txt' % self.lang
            train_file = '../training_data_%s.txt' % self.lang

            (x, y)= parser.fold_split(data_file, test_file, j+1)
            print "Number of test sentences: %d" % y

            splitter = DataParser()
            splitter.load_data(data_file)
            if small_dataset:
                for i in [1,3,5]:
                    splitter.choose_training_data_random(train_file, i)
                    (dep_accuracy, arc_accuracy) = self.single_experiment(train_file, test_file, mode, k)

                    dep_total[i] += dep_accuracy
                    arc_total[i] += arc_accuracy

                    if not incremental:
                        splitter.reset_splits()
            step = 10
            if exponential:
                step = 1
            for i in range(range_start, range_end+1, step):
                if not exponential:
                    train_num = int((i/100.0) * total_data)
                else:
                    train_num = 2**i
                # print "Training instances: %d" % train_num
                if self.conll_format:
                    d = splitter.choose_training_data(train_file, train_num)
                else:
                    d = splitter.choose_training_data_random(train_file, train_num)
                # print "Training instances chosen: %d" % d
                (dep_accuracy, arc_accuracy) = self.single_experiment(train_file, test_file, mode, k)

                dep_total[i] += dep_accuracy
                arc_total[i] += arc_accuracy

                if not incremental:
                    splitter.reset_splits()

        for key in sorted(dep_total.keys()):

            if key < 10:
                train_str = "%d" % key
            else:
                train_str = "%d%%" % key
            print "%2.3f\t%2.3f" \
                      % (dep_total[key]/folds, arc_total[key]/folds)

    def single_experiment(self, train_file, test_file, mode, k=1):
        fv_file = '../training.dat'
        train_data, morph_features = self.extractor.get_sentences(train_file)
        # print [d[0] for d in train_data]
        # print "Training data size: %d" % len(train_data)
        classifier, morph_features = self.train(train_data, morph_features, fv_file, model=mode, k=k)
        test_data, max_test_features = self.extractor.get_sentences(test_file, morph_feats=morph_features)
        (predictions, real_dependencies) = self.predict(test_data, morph_features, classifier)
        # print predictions
        accuracy = self.get_raw_accuracy(predictions, real_dependencies)
        return accuracy
        # return (0.0, 0.0)



