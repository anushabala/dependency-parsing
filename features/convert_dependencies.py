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
# todo: Add methods for LA, RA, S instead of just hardcoding in what they do
#todo: add object serialization - when a model is written, the set of features is serialized and saved, and a file that
# maintains the mapping between .dat file and serialized object is maintained
# todo: WRITE OUT THE FV MAPPINGS - this is needed to actually do classification
# todo: clean up generally
import datetime

columns = {"index": 0, "word": 1, "stem": 2, "morph": 3, "pos": 4, "head": 5, "dep": 6}
actions = {"S": 1, "LA": 2, "RA": 3, "END": 4}
training_fvs = []
LABEL = "dep"


def get_property(properties, pos, propName):
    relevant_props = properties[pos]
    if relevant_props[columns[propName] - 1] == '':
        return "0"
    if propName == "head":
        return int(relevant_props[columns[propName] - 1])
    return relevant_props[columns[propName] - 1]

def set_property(properties, pos, propName, new_value):
    relevant_props = properties[pos]
    # while columns[propName] - 1 <= len(relevant_props):
    #     relevant_props.append('')
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
                line = [("%d:%s" % (f, v)) for (f, v) in vectors.iteritems()]
                line = "\t".join(line)
                training_fvs.append((action_num, dep_num, line))

        return training_fvs

    def write_fv_mappings(self, filepath):
        mapping_file = open(filepath, 'wb')
        print self.FV_MAPPINGS
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
        self.model = None
        self.k = k

    def add_training_data(self, states):
        self.extractor.add_fv_mappings(states)
        self.training_data.append(states)

    def write_training_data(self, filepath, fvs):
        train_file = open(filepath, 'w')
        for (action, dep, vectors) in fvs:
            line = "%s\t%s\t%s\n" % (str(action), str(dep), vectors)
            train_file.write(line)
        train_file.close()
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')

        model_file = open(self.model_path % st, 'wb')
        pickle.dump(fvs, model_file)
        model_file.close()

        self.extractor.write_fv_mappings(self.fv_mapping_path % st)

        if os.path.isfile(self.mapping_path):
            mapping_file = open(self.mapping_path, 'rb')
            file_map = pickle.load(mapping_file)
            mapping_file.close()
            file_map[train_file] = (self.model_path % st, self.fv_mapping_path % st)
            mapping_file = open(self.mapping_path, 'wb')
            pickle.dump(file_map, mapping_file)
            mapping_file.close()
        else:
            file_map = {filepath: (self.model_path % st, self.fv_mapping_path % st)}
            print file_map
            mapping_file = open(self.mapping_path, 'wb')
            pickle.dump(file_map,mapping_file)
            mapping_file.close()

        # return (self.model_path % st, self.fv_mapping_path % st)

    def train(self, filepath='../training.dat'):
        training_fvs = self.extractor.convert_to_fvs(self.training_data)
        self.model = training_fvs
        self.write_training_data(filepath, training_fvs)

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


    def get_next_action(self, state):
        min_distances = []
        fv_state = self.extractor.convert_instance_to_fv(state)
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
        # print smallest
        next_action = defaultdict(int)

        next_dep = defaultdict(int)
        for (dist, pos) in smallest:
            fv = training_fvs[pos]
            action_num = fv[0]
            dep_num = fv[1]
            next_action[action_num] += 1
            next_dep[dep_num] += 1

        chosen_action = max(next_action.iteritems(), key=operator.itemgetter(1))[0]
        chosen_dep = max(next_dep.iteritems(), key=operator.itemgetter(1))[0]

        if chosen_action == actions["LA"] and len(self.S)==0:
            chosen_action = actions["S"]
        if chosen_action == actions["RA"] and len(self.S)==0:
            chosen_action = actions["S"]

        if chosen_action == actions["S"] and chosen_dep != 0:
            chosen_dep = 0
        elif chosen_action != actions["S"] and chosen_dep == 0:
            chosen_dep = \
                max([(key, value) for (key, value) in next_dep.iteritems() if value != 0], key=operator.itemgetter(1))[0]


        # print "%d\t%d" %(chosen_action,chosen_dep)
        return (chosen_action, chosen_dep)
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
                self.S.append(self.I[0])
                self.I = self.I[1:]
            else:
                stack_top = self.S[-1]
                stack_head = get_property(properties, stack_top, "head")
                input_head = get_property(properties, self.I[0], "head")
                if stack_head == self.I[0]:
                    self.S.pop()
                    dep = get_property(properties, stack_top, "dep")
                    features.append(("LA", dep, state))
                    self.A.append((stack_head, dep, stack_top))
                elif input_head == stack_top:
                    dep = get_property(properties, self.I[0], "dep")
                    features.append(("RA", dep, state))
                    self.A.append((input_head, dep, self.I[0]))
                    self.S.append(self.I[0])
                    self.I = self.I[1:]

                else:
                    features.append(("S", 'NULL', state))
                    self.S.append(self.I[0])
                    self.I = self.I[1:]
        #final state
        state = self.get_current_state(properties)
        features.append(("END", 'NULL', state))

        for (state, dep, feat) in features:
            feat = [self.mappings[f] if f in self.mappings.keys() else f for f in feat]
            # print feat
            # print "%s\t%s\t%s" % (state, dep, "\t".join(feat))
            modified_features.append((state, dep, feat))
        self.A = [(self.mappings[h], l, self.mappings[d]) for (h, l, d) in self.A]
        # print "Dependencies: %s\n" % self.A

        return modified_features

    def predict_actions(self, sentence, properties):
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
            fv = convert_instance_to_fv(lex_state)
            (action, dep) = self.predict_next_action(fv) #todo: return actual dep value instead of vector
            features.append((action, dep, state))
            if action == actions["S"]:
                self.S.append(self.I[0])
                self.I = self.I[1:]

            elif action == actions["LA"]:
                stack_top = self.S[-1]
                self.S.pop()
                set_property(properties, stack_top, "head", self.I[0])
                set_property(properties, stack_top, "dep", FV_MAPPINGS[LABEL][dep])

                self.A.append((self.I[0], dep, stack_top))

            elif action == actions["RA"]:
                stack_top = self.S[-1]
                set_property(properties, self.I[0], "head", stack_top)
                set_property(properties, self.I[0], "dep", FV_MAPPINGS[LABEL][dep])

                self.A.append((stack_top, dep, self.I[0]))
                self.S.append(self.I[0])
                self.I = self.I[1:]

        state = self.get_current_state(properties)
        features.append(("END", 'NULL', state))

        for (action, dep, state) in features:
            state = [self.mappings[f] if f in self.mappings.keys() else f for f in state]
            # print feat
            # print "%s\t%s\t%s" % (state, dep, "\t".join(feat))
            # print len(feat)
            modified_features.append((action, dep, state))

        # modified_A = [(self.mappings[h], FV_MAPPINGS[LABEL][l], self.mappings[d]) for (h, l, d) in self.A]
        self.A = [(h, FV_MAPPINGS[LABEL][l], d) for (h, l, d) in self.A]
        self.predict_root(sentence)

        return self.A

    def predict_root(self, sentence):
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

    def predict_next_action(self, test_vectors):
        min_distances = []
        for j in range(0, len(training_fvs)):
            f = training_fvs[j]
            train_vectors = f[2]
            train_dist = 0.0
            for i in range(0, len(train_vectors)):
                train_v = train_vectors[i]
                test_v = test_vectors[i]
                dist = 0.0
                for c in range(0, len(train_v)):
                    diff = math.fabs(int(train_v[c]) - int(test_v[c])) ** 2
                    dist += diff
                dist = math.sqrt(dist)
                train_dist += dist
            heappush(min_distances, (train_dist, j))
        smallest = nsmallest(self.k, min_distances)
        # print smallest
        next_action = defaultdict(int)

        next_dep = defaultdict(int)
        for (dist, pos) in smallest:
            fv = training_fvs[pos]
            action_num = fv[0]
            dep_num = fv[1]
            next_action[action_num] += 1
            next_dep[dep_num] += 1

        chosen_action = max(next_action.iteritems(), key=operator.itemgetter(1))[0]
        chosen_dep = max(next_dep.iteritems(), key=operator.itemgetter(1))[0]

        if chosen_action == actions["LA"] and len(self.S)==0:
            chosen_action = actions["S"]
        if chosen_action == actions["RA"] and len(self.S)==0:
            chosen_action = actions["S"]

        if chosen_action == actions["S"] and chosen_dep != 0:
            chosen_dep = 0
        elif chosen_action != actions["S"] and chosen_dep == 0:
            chosen_dep = \
                max([(key, value) for (key, value) in next_dep.iteritems() if value != 0], key=operator.itemgetter(1))[0]


        # print "%d\t%d" %(chosen_action,chosen_dep)
        return (chosen_action, chosen_dep)


    def reset(self):
        self.S = []
        self.I = []
        self.A = []  # the list of dependents on any given token
        self.mappings = {}


def add_fv_mappings(states):
    global FV_MAPPINGS
    for f in states:
        state = f[2]
        dep = f[1]
        if dep not in FV_MAPPINGS[LABEL]:
            FV_MAPPINGS[LABEL].append(dep)
        for i in range(0, len(state)):
            if state[i] not in FV_MAPPINGS[i]:
                FV_MAPPINGS[i].append(state[i])


def convert_to_fvs(all_features, fv_file):
    global training_fvs
    training_fvs = []
    train_file = open(fv_file, 'w')
    for sent_features in all_features:
        for (action_name, dep, state) in sent_features:
            action_num = actions[action_name]
            dep_num = FV_MAPPINGS[LABEL].index(dep)
            vectors = convert_instance_to_fv(state)
            line = [("%d:%s" % (f, v)) for (f, v) in vectors.iteritems()]
            line = "\t".join(line)
            train_file.write("%s\t%s\t%s\n" % (str(action_num), str(dep_num), line))
            training_fvs.append((action_num, dep_num, vectors))
    train_file.close()


def convert_instance_to_fv(state):
    vectors = defaultdict(str)
    for i in range(0, len(state)):
        v = ["0"] * len(FV_MAPPINGS[i])
        if state[i] in FV_MAPPINGS[i]:
            v[FV_MAPPINGS[i].index(state[i])] = "1"
        v = "".join(v)
        vectors[i] = v
    return vectors


def train(filepath, max=-10, start=1, print_status=False):
    # print("[Training]")
    global FV_MAPPINGS
    FV_MAPPINGS = defaultdict(lambda: ["NULL"])
    training_features = []
    parser = Parser()
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

            # get features here
            parser.reset()
            num += 1
            if num >= start:
                sent_features = parser.get_state_sequence(sentence, properties)
                add_fv_mappings(sent_features)
                training_features.append(sent_features)
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

    convert_to_fvs(training_features, "../training.dat")
    infile.close()
    # print "Completed training"


def get_raw_accuracy(predictions, real_properties):
    test_dependencies = {}
    for key in predictions.keys():
        dependencies = real_properties[key]
        all_deps = []
        for index in dependencies:
            head = get_property(dependencies, index, "head")
            label = get_property(dependencies, index, "dep")
            all_deps.append((head, label, index))
        test_dependencies[key] = all_deps

    # print test_dependencies
    # print predictions
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

def predict(filepath, max=-3.14, start=1, print_status=False, k=1):
    # print("[Testing]")
    parser = Parser(k)
    infile = open(filepath, 'r')
    line = infile.readline()
    first = True
    testing = False
    properties = defaultdict(list)
    test_properties = defaultdict(list)
    tested_sentences = []
    sentence = None
    num = 0
    real_dependencies = {}
    predictions = {}
    while line:
        line = line.strip()
        line = line.lower()
        if line == "":

            # get features here
            parser.reset()
            num += 1
            if num >= start:
                tested_sentences.append(" ".join(sentence))
                sent_pred = parser.predict_actions(sentence, properties)
                predictions[num - 1] = sent_pred
                real_dependencies[num - 1] = test_properties
                testing = True
                if testing and print_status:
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

    return (predictions, real_dependencies)

def incremental_train(filepath):
    for i in range(1, 75, 5):
        k = max(i/2 - 1, 1)
        train_start = 1
        train_num = i
        test_start = train_start + train_num
        train(filepath, train_num, train_start)
        (predictions, real_dependencies) = predict(filepath, start=test_start, k=k, )
        accuracy = get_raw_accuracy(predictions, real_dependencies)
        print("Training set size: %d\tk: %d\tAccuracy: %2.3f" % (train_num, k, accuracy))


def single_experiment(filepath):
    train_start = 1
    train_num = 1
    test_start = train_start + train_num
    test_num = -1
    #
    train('../welt-annotation-spatial.txt', train_num, train_start)
    (predictions, real_dependencies) = predict('../welt-annotation-spatial.txt', max=test_num, start=test_start)
    accuracy = get_raw_accuracy(predictions, real_dependencies)
    print accuracy

def new_design(filepath):
    start = 1
    max = 10
    classifier = ParseClassifier()
    parser = Parser()
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

            # get features here
            parser.reset()
            num += 1
            if num >= start:
                sent_features = parser.get_state_sequence(sentence, properties)
                classifier.add_training_data(sent_features)
                training = True
                if training:
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

    classifier.train()
    infile.close()
    # print "Completed training"

# new_design('../welt-annotation-spatial.txt')
classifier = ParseClassifier()
classifier.load_model("../training.dat")

