# Written by: Anusha Balakrishnan
# Date: 10/6/14
from collections import defaultdict
from random import randint
from heapq import heappush, nsmallest
import math
import operator
import pickle
# Use Weka - try different classification algs for the dependency labels, for parser action
# start using differeent method when you have more training data?
#
# todo: handle cases where LA or RA is predicted but can't occur - count as misclassification?
# todo: write model and training data separately

import datetime
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


columns = {"index": 0, "word": 1, "stem": 2, "morph": 3, "pos": 4, "head": 5, "dep": 6}
actions = {"S": 1, "LA": 2, "RA": 3, "END": 4}
training_fvs = []


def get_property(properties, pos, propName):
    relevant_props = properties[pos]
    if relevant_props[columns[propName] - 1] == '':
        return "0"
    if propName == "head":
        return int(relevant_props[columns[propName] - 1])
    return relevant_props[columns[propName] - 1]

def set_property(properties, pos, propName, new_value):
    relevant_props = properties[pos]
    relevant_props[columns[propName] - 1] = new_value

class DataParser:
    def __init__(self):
        self.all_data = []
    def write_data(self, fvs, filepath):
        out_file = open(filepath, 'w')
        for (action, dep, vectors) in fvs:
            vecs = [("%d:%s" % (f, v)) for (f, v) in vectors.iteritems()]
            vecs = "\t".join(vecs)
            line = "%s\t%s\t%s\n" % (str(action), str(dep), vecs)
            out_file.write(line)
        out_file.close()

    def load_data(self, filepath):
        # print("[Testing]")

        infile = open(filepath, 'r')
        line = infile.readline()
        properties = []
        while line:
            line = line.strip()
            line = line.lower()
            if line == "":
                self.all_data.append(properties)
                properties = []
            else:
                line = line.strip()
                properties.append(line)
            line = infile.readline()

        infile.close()

    def random_split(self, train_file, test_file, ratio):
        train_file = open(train_file, 'w')
        test_file = open(test_file, 'w')
        train_num = int(ratio * len(self.all_data))
        current = 0
        train_data = []
        test_data = []
        while current<train_num:
            pos = randint(current, len(self.all_data)-1)
            train_data.append(self.all_data[pos])
            self.all_data[pos], self.all_data[current] = self.all_data[current], self.all_data[pos]
            current += 1

        while current<len(self.all_data):
            test_data.append(self.all_data[current])
            current+=1


        for sentence in train_data:
            for line in sentence:
                train_file.write(line+"\n")
            train_file.write("\n")
        for sentence in test_data:
            for line in sentence:
                test_file.write(line+"\n")
            test_file.write("\n")

        train_file.close()
        test_file.close()

        return (len(train_data), len(test_data))



class FeatureExtractor:
    def __init__(self):
        self.FV_MAPPINGS = defaultdict(list)
        self.LABEL = "dep"
        self.FV_MAPPINGS[self.LABEL] = ["NULL"]

    def add_fv_mappings(self, states):
        for f in states:
            state = f[2]
            dep = f[1]

            if dep not in self.FV_MAPPINGS[self.LABEL]:
                self.FV_MAPPINGS[self.LABEL].append(dep)
            for i in range(0, len(state)):
                if i not in self.FV_MAPPINGS.keys():
                    self.FV_MAPPINGS[i] = ["NULL"]
                if state[i] not in self.FV_MAPPINGS[i]:
                    self.FV_MAPPINGS[i].append(state[i])

    def convert_instance_to_fv(self, state):
        vectors = defaultdict(str)
        for i in range(0, len(state)):
            v = ["0"] * len(self.FV_MAPPINGS[i])
            if state[i] in self.FV_MAPPINGS[i]:
                v[self.FV_MAPPINGS[i].index(state[i])] = "1"
            v = "".join(v)
            vectors[i] = v
        return vectors

    def convert_to_fvs(self, all_features):
        training_fvs = []
        for sent_features in all_features:
            for (action_name, dep, state) in sent_features:
                action_num = actions[action_name]
                dep_num = self.FV_MAPPINGS[self.LABEL].index(dep)
                vectors = self.convert_instance_to_fv(state)
                training_fvs.append((action_num, dep_num, vectors))

        return training_fvs

    def write_fv_mappings(self, filepath):
        mapping_file = open(filepath, 'wb')
        pickle.dump(self.FV_MAPPINGS, mapping_file)
        mapping_file.close()

    def load_fv_mappings(self, filepath):
        mapping_file = open(filepath, 'rb')
        self.FV_MAPPINGS = pickle.load(mapping_file)
        mapping_file.close()

    def convert_to_values(self, values):
        new_val = []
        for f in values:
            pos = 0
            if '1' in f:
                pos = f.index('1') + 1
            new_val.append(pos)

        return new_val

class ParseClassifier:
    def __init__(self, k=1, mode="knn"):
        self.extractor = FeatureExtractor()
        self.training_data = []
        self.action_classifier = None
        self.dep_classifier = None
        self.k = k
        self.mode = mode

    def get_training_data(self):
        return self.training_data

    def train(self, data):
        self.add_fv_mappings(data)
        training_fvs = self.extractor.convert_to_fvs(data)
        self.training_data = training_fvs
        if self.mode=="svm":
            self.__train_svm()
            # self.write_training_data(filepath, training_fvs, model=trained_svm)
        elif self.mode=="decision_tree":
            self.__train_decision_tree()
        else:
            self.__train_knn()

    def add_fv_mappings(self, data):
        for state in data:
            self.extractor.add_fv_mappings(state)

    def write_fv_mappings(self, filepath):
        self.extractor.write_fv_mappings(filepath)


    def write_model(self, filepath, dep_classifier_path=None):
        if self.mode=="knn":
            pickle.dump(self.training_data, filepath)
        elif self.mode=="svm":
            pickle.dump(self.action_classifier, filepath)
            pickle.dump(self.dep_classifier, dep_classifier_path)
        elif self.mode=="linear_svm":
            pickle.dump(self.action_classifier, filepath)
            pickle.dump(self.dep_classifier, dep_classifier_path)

    def __train_svm(self):
        self.action_classifier = SVC(kernel='rbf')
        self.dep_classifier = SVC(kernel='rbf')
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
        if self.mode=="knn":
            self.training_data = pickle.load(filepath)
        elif self.mode=="linear_svm":
            self.action_classifier = pickle.load(filepath)
            self.dep_classifier = pickle.load(dep_classifier_path)
        elif self.mode=="decision_tree":
            self.action_classifier = pickle.load(filepath)
            self.dep_classifier = pickle.load(dep_classifier_path)

    def get_next_action(self, state):

        fv_state = self.extractor.convert_instance_to_fv(state)
        if self.mode=="svm":
            (chosen_action, chosen_dep) = self.__svm(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])
        elif self.mode=="decision_tree":
            (chosen_action, chosen_dep) = self.__decision_tree(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])
        elif self.mode=="knn":
            (chosen_action, chosen_dep) = self.__sklearn_knn(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])

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

class Parser:
    def __init__(self, k=1):
        self.S = []
        self.I = []
        self.A = []  # the list of dependents on any given token
        self.mappings = {}



    def get_current_state(self, properties):
        features = []
        # Feature 1: POS:stack(1), morph:stack(1)
        if len(self.S) > 1:
            features.append(get_property(properties, self.S[-2], "pos"))
            features.append(get_property(properties, self.S[-2], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")

        # POS:stack(0)
        if len(self.S) > 0:
            features.append(get_property(properties, self.S[-1], "pos"))
            features.append(get_property(properties, self.S[-1], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")

        # POS: input(0)
        if len(self.I) > 0:
            features.append(get_property(properties, self.I[0], "pos"))
            features.append(get_property(properties, self.I[0], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
        # POS: input(1)
        if len(self.I) > 1:
            features.append(get_property(properties, self.I[1], "pos"))
            features.append(get_property(properties, self.I[1], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
        # POS: input(0)
        if len(self.I) > 2:
            features.append(get_property(properties, self.I[2], "pos"))
            features.append(get_property(properties, self.I[2], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
        # POS: input(0)
        if len(self.I) > 3:
            features.append(get_property(properties, self.I[3], "pos"))
            features.append(get_property(properties, self.I[3], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")

        if len(self.S) > 0:
            top_pos = self.S[-1]
            leftmost = top_pos
            left_label = "NULL"
            rightmost = top_pos
            right_label = "NULL"
            dep_0 = "NULL"
            for (head, label, dep) in self.A:
                if dep == top_pos and dep_0 != None:
                    dep_0 = label
                elif head == top_pos:
                    if dep < leftmost:
                        leftmost = dep
                        left_label = label
                    elif dep > rightmost:
                        rightmost = dep
                        right_label = label
            #DEP:stack(0)
            features.append(dep_0)
            #Leftmost dependent of stack(0)
            features.append(left_label)
            #rightmost dependent of stack(0)
            features.append(right_label)
        else:
            features.append("NULL")
            features.append("NULL")
            features.append("NULL")

        #leftmost dependent of current input token
        if len(self.I) > 0:
            input_pos = self.I[0]
            left_dep = "NULL"
            leftmost = input_pos
            for (head, label, dep) in self.A:
                if head == input_pos:
                    if dep < leftmost:
                        leftmost = dep
                        left_dep = label
            features.append(left_dep)
        else:
            features.append("NULL")

        if len(self.S) > 0:
            top = self.S[-1]
            #word at the top of the stack
            top_word = get_property(properties, top, "word")
            features.append(top_word)
            top_head = "NULL"
            for (head, label, dep) in self.A:
                if dep == top:
                    top_head = head
                    break
            #head of word at stack top
            features.append(top_head)

        else:
            features.append("NULL")
            features.append("NULL")

        #current input word
        if len(self.I) > 0:
            features.append(self.I[0])
        else:
            features.append("NULL")
        #next input word
        if len(self.I) > 1:
            features.append(self.I[1])
        else:
            features.append("NULL")

        return features

    def get_state_sequence(self, sentence, properties):
        features = []
        modified_features = []
        self.S = []
        self.I = [i for i in range(0, len(sentence))]
        positions = [(self.I[i], sentence[i]) for i in range(0, len(sentence))]
        self.mappings = dict(positions)
        self.A = []  # the list of dependents on any given token
        i = 0
        while len(self.I) > 0:
            state = self.get_current_state(properties)

            if len(self.S) == 0:
                features.append(("S", 'NULL', state))
                self.shift()

            else:
                stack_top = self.S[-1]
                stack_head = get_property(properties, stack_top, "head")
                input_head = get_property(properties, self.I[0], "head")
                if stack_head == self.I[0]:

                    dep = get_property(properties, stack_top, "dep")
                    features.append(("LA", dep, state))
                    self.left_arc(stack_head, dep, stack_top)
                elif input_head == stack_top:
                    dep = get_property(properties, self.I[0], "dep")
                    features.append(("RA", dep, state))
                    self.right_arc(input_head, dep)

                else:
                    features.append(("S", 'NULL', state))
                    self.shift()

        #final state
        state = self.get_current_state(properties)
        features.append(("END", 'NULL', state))

        for (state, dep, feat) in features:
            feat = [self.mappings[f] if f in self.mappings.keys() else f for f in feat]
            modified_features.append((state, dep, feat))
        self.A = [(self.mappings[h], l, self.mappings[d]) for (h, l, d) in self.A]

        return modified_features

    def left_arc(self, head, label, dep):
        self.S.pop()
        self.A.append((head, label, dep))
    def right_arc(self, head, label):
        self.A.append((head, label, self.I[0]))
        self.S.append(self.I[0])
        self.I = self.I[1:]

    def shift(self):
        self.S.append(self.I[0])
        self.I = self.I[1:]

    def predict_actions(self, sentence, properties, classifier):
        # print "extracting"
        features = []
        modified_features = []
        self.S = []
        self.I = [i for i in range(0, len(sentence))]
        positions = [(self.I[i], sentence[i]) for i in range(0, len(sentence))]
        self.mappings = dict(positions)
        self.A = []  # the list of dependents on any given token
        i = 0
        while len(self.I) > 0:
            state = self.get_current_state(properties)
            lex_state = [self.mappings[f] if f in self.mappings.keys() else f for f in state]
            (action, dep) = classifier.get_next_action(lex_state)
            # print "predicted action %s dep %s" % (str(action), str(dep))
            if (action==actions["LA"] or action==actions["RA"]) and len(self.S) == 0:
                action = actions["S"]
            features.append((action, dep, state))
            if action == actions["S"]:
                self.shift()

            elif action == actions["LA"]:
                stack_top = self.S[-1]
                set_property(properties, stack_top, "head", self.I[0])
                set_property(properties, stack_top, "dep", dep)
                self.left_arc(self.I[0], dep, stack_top)

            elif action == actions["RA"]:
                stack_top = self.S[-1]
                set_property(properties, self.I[0], "head", stack_top)
                set_property(properties, self.I[0], "dep", dep)

                self.right_arc(stack_top, dep)
            elif action == actions["END"]:
                break

        state = self.get_current_state(properties)
        features.append(("END", 'NULL', state))

        for (action, dep, state) in features:
            state = [self.mappings[f] if f in self.mappings.keys() else f for f in state]
            modified_features.append((action, dep, state))

        self.find_root(sentence)

        return self.A

    def find_root(self, sentence):
        positions = [i for i in range(0, len(sentence))]
        root = -1
        for i in positions:
            is_dep = False
            for (head, label, dep) in self.A:
                if i == dep:
                    is_dep = True
                    break
            if not is_dep:
                root = i
                break

        self.A.append((0, 'root', root))



    def reset(self):
        self.S = []
        self.I = []
        self.A = []  # the list of dependents on any given token
        self.mappings = {}



def train(filepath, train_file, print_status=False, model="knn"):
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

    classifier.train(training_data)
    training_fvs = classifier.get_training_data()
    writer = DataParser()
    writer.write_data(training_fvs, train_file)

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

def predict(filepath, model_path=None, print_status=False, k=1, classifier=None, mode="knn"):
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
    for i in range(10, 91, 10):
        (dep_accuracy, arc_accuracy) = cross_validate(filepath, i, mode, k, folds)
        print "%d%%\t%2.3f\t%2.3f" \
              % (i, dep_accuracy, arc_accuracy)

def cross_validate(filepath, i, mode, k=1, folds=5):
    parser = DataParser()
    parser.load_data(filepath)
    train_file = '../training_data.txt'
    test_file = '../test_data.txt'
    train_fvs = '../training.dat'
    dep_total = 0.0
    arc_total = 0.0
    for j in range(0, folds):
        (num_train, num_test) = parser.random_split(train_file, test_file, i/100.0)
        classifier = train(train_file, train_fvs, model=mode)
        if k<0:
            k = max(int(num_train/2) - 1, 1)

        (predictions, real_dependencies) = predict(test_file, k=k, classifier=classifier, mode=mode)
        (dep_accuracy, arc_accuracy) = get_raw_accuracy(predictions, real_dependencies)
        dep_total += dep_accuracy
        arc_total += arc_accuracy

    return (dep_total/folds, arc_total/folds)


def single_experiment(filepath):

    classifier = train('../welt-annotation-spatial.txt', '../training.dat', model="svm")
    (predictions, real_dependencies) = predict('../welt-annotation-spatial.txt', '../training.dat', classifier=classifier)

    accuracy = get_raw_accuracy(predictions, real_dependencies)
    print accuracy


# predict('../welt-annotation-spatial.txt', start=11, max=1 print_status=True, k=4)
# incremental_train('../welt-annotation-spatial.txt', "decision_tree", k=5, folds=10)
