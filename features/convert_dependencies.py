# Written by: Anusha Balakrishnan
# Date: 10/6/14
from collections import defaultdict
import os
import time
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
from sklearn import svm


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

class ParseClassifier:
    def __init__(self, k=1):
        self.mapping_path = "../models/mapping.dat"
        self.fv_mapping_path = "../models/fv_mapping%s.dat"
        self.model_path = "../models/model%s."
        self.extractor = FeatureExtractor()
        self.training_data = []
        self.action_classifier = None
        self.dep_classifier = None
        self.model = None
        self.k = k
        self.classifiers = { "knn": self.__knn,
                             "linear_svm": self.__linear_svm}
        self.train_funs = {"svm": self.__train_svm}


    def add_training_data(self, states):
        self.extractor.add_fv_mappings(states)
        self.training_data.append(states)

    def write_training_data(self, filepath, fvs, model=None):
        train_file = open(filepath, 'w')
        for (action, dep, vectors) in fvs:
            vecs = [("%d:%s" % (f, v)) for (f, v) in vectors.iteritems()]
            vecs = "\t".join(vecs)
            line = "%s\t%s\t%s\n" % (str(action), str(dep), vecs)
            train_file.write(line)
        train_file.close()
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')

        model_file = open(self.model_path % st, 'wb')
        if model==None:
            pickle.dump(fvs, model_file)
        else:
            pickle.dump(model, model_file)
        model_file.close()

        self.extractor.write_fv_mappings(self.fv_mapping_path % st)

        if os.path.isfile(self.mapping_path):
            mapping_file = open(self.mapping_path, 'rb')
            file_map = pickle.load(mapping_file)
            mapping_file.close()
            file_map[filepath] = [self.model_path % st, self.fv_mapping_path % st]

            mapping_file = open(self.mapping_path, 'wb')
            pickle.dump(file_map, mapping_file)
            mapping_file.close()
        else:
            file_map = {filepath: [self.model_path % st, self.fv_mapping_path % st]}
            mapping_file = open(self.mapping_path, 'wb')
            pickle.dump(file_map,mapping_file)
            mapping_file.close()


    def train(self, filepath='../training.dat', mode="knn"):
        training_fvs = self.extractor.convert_to_fvs(self.training_data)
        self.model = training_fvs
        if mode=="svm":
            self.__train_svm()
            # self.write_training_data(filepath, training_fvs, model=trained_svm)
        else:
            self.write_training_data(filepath, training_fvs)




    def __train_svm(self):
        self.action_classifier = svm.SVC()
        self.dep_classifier = svm.SVC()
        action_labels = [f[0] for f in self.model]
        dep_labels = [f[1] for f in self.model]
        fvs = [f[2].values() for f in self.model]
        self.action_classifier.fit(fvs, action_labels)
        self.dep_classifier.fit(fvs, dep_labels)

    def load_model(self, filepath):
        if not os.path.isfile(self.mapping_path):
            print "Error: could not find mapping file for previously saved model and feature vectors"
        else:
            mapping_file = open(self.mapping_path, 'rb')
            file_map = pickle.load(mapping_file)
            mapping_file.close()
            if filepath not in file_map.keys():
                print "[Error: Could not find model associated with training data at %s]" % filepath
                return
            (model_file, fv_file) = file_map[filepath]
            mfile = open(model_file, 'rb')
            self.model = pickle.load(mfile)
            mfile.close()

            self.extractor.load_fv_mappings(fv_file)


    def get_next_action(self, state, mode="knn"):
        fv_state = self.extractor.convert_instance_to_fv(state)
        if mode=="linear_svm":
            (chosen_action, chosen_dep) = self.__linear_svm(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])
        elif mode=="knn":
            (chosen_action, chosen_dep) = self.__knn(fv_state)
            return (chosen_action, self.extractor.FV_MAPPINGS[self.extractor.LABEL][chosen_dep])


    def __knn(self, fv_state):
        min_distances = []
        for j in range(0, len(self.model)):
            f = self.model[j]
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
            fv = self.model[pos]
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

    def __linear_svm(self, fv_state):

        pred_action = self.action_classifier.predict(fv_state.values())[0]
        pred_dep = self.dep_classifier.predict(fv_state.values())[0]

        return (pred_action, pred_dep)

class Parser:
    def __init__(self, k=1):
        self.S = []
        self.I = []
        self.A = []  # the list of dependents on any given token
        self.mappings = {}
        self.k = k


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
            (action, dep) = classifier.get_next_action(lex_state, mode="linear_svm")
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



def train(filepath, train_file, max=-10, start=1, print_status=False):
    # print("[Training]")
    global FV_MAPPINGS
    FV_MAPPINGS = defaultdict(lambda: ["NULL"])
    training_features = []
    parser = Parser()
    classifier = ParseClassifier()
    infile = open(filepath, 'r')
    line = infile.readline()
    first = True
    training = False
    properties = {}
    sentence = None
    num = 0

    while line:
        line = line.strip()
        line = line.lower()
        if line == "":

            parser.reset()
            num += 1
            if num >= start:
                sent_features = parser.get_state_sequence(sentence, properties)
                classifier.add_training_data(sent_features)
                training = True
                if training and print_status:
                    print "%d:\t%s" % (num, sentence)


            if num == max + start - 1:
                break

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

    classifier.train(train_file, mode="svm")
    infile.close()
    return classifier
    # print "Completed training"


def get_raw_accuracy(predictions, test_dependencies):
    print predictions
    print test_dependencies
    num = 0
    total = 0.0
    correct = 0.0

    for key in test_dependencies.keys():
        for (head, label, dep) in test_dependencies[key]:
            total += 1
            if (head, label, dep) in predictions[key]:
                correct += 1

    accuracy = correct / total
    return accuracy

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

def predict(filepath, model_path=None, max=-3.14, start=1, print_status=False, k=1, classifier=None):
    # print("[Testing]")
    parser = Parser(k)
    if classifier==None:
        classifier = ParseClassifier()
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
            if num >= start:
                tested_sentences.append(" ".join(sentence))
                sent_pred = parser.predict_actions(sentence, properties, classifier)
                predictions[num - 1] = sent_pred
                real_properties[num - 1] = test_properties
                if print_status:
                    print "%d:\t%s" % (num, sentence)


            if num == max + start - 1:
                break
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

def incremental_train(filepath):
    train_file = '../training.dat'
    for i in range(1, 25, 5):
        k = 1
        train_start = 1
        train_num = i
        test_start = train_start + train_num
        train(filepath, train_file, train_num, train_start)
        (predictions, real_dependencies) = predict(filepath, train_file, start=test_start, k=k, )
        accuracy = get_raw_accuracy(predictions, real_dependencies)
        print("Training set size: %d\tk: %d\tAccuracy: %2.3f" % (train_num, k, accuracy))


def single_experiment(filepath):
    train_num = 10
    train_start=1
    test_start = train_start + train_num
    test_num = 5
    #
    classifier = train('../welt-annotation-spatial.txt', '../training.dat', train_num, train_start)
    (predictions, real_dependencies) = predict('../welt-annotation-spatial.txt', '../training.dat', max=test_num, start=test_start, classifier=classifier)

    accuracy = get_raw_accuracy(predictions, real_dependencies)
    print accuracy


# new_design('../welt-annotation-spatial.txt')
# predict('../welt-annotation-spatial.txt', start=11, max=1 print_status=True, k=4)

single_experiment('../welt-annotation-spatial.txt')
# incremental_train('../welt-annotation-spatial.txt')