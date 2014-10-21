# Written by: Anusha Balakrishnan
#Date: 10/6/14
from collections import defaultdict
from heapq import heappush, nsmallest
import math
import operator

columns = {"index":0, "word":1, "stem":2, "morph":3, "pos":4, "head":5, "dep":6}
actions = {"S":1, "LA":2, "RA":3, "END":4}
training_fvs = []
LABEL = "dep"
FV_MAPPINGS = defaultdict(lambda : ["NULL"])
k = 1
class Parser:
    def __init__(self):
        self.S = []
        self.I = []
        self.A = [] # the list of dependents on any given token
        self.mappings = {}

    def get_property(self, properties, pos, propName):
        relevant_props = properties[pos]
        if relevant_props[columns[propName] -1] == '':
            return "0"
        if propName=="head":
            return int(relevant_props[columns[propName] -1])
        return relevant_props[columns[propName] -1]
    def get_action_features(self, properties):
        features = []
        # Feature 1: POS:stack(1), morph:stack(1)
        if len(self.S)>1:
            features.append(self.get_property(properties, self.S[-2], "pos"))
            features.append(self.get_property(properties, self.S[-2], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
    
        # POS:stack(0)
        if len(self.S)>0:
            features.append(self.get_property(properties, self.S[-1], "pos"))
            features.append(self.get_property(properties, self.S[-1], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
    
        # POS: input(0)
        if len(self.I) > 0:
            features.append(self.get_property(properties, self.I[0], "pos"))
            features.append(self.get_property(properties, self.I[0], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
        # POS: input(1)
        if len(self.I) > 1:
            features.append(self.get_property(properties, self.I[1], "pos"))
            features.append(self.get_property(properties, self.I[1], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
        # POS: input(0)
        if len(self.I) > 2:
            features.append(self.get_property(properties, self.I[2], "pos"))
            features.append(self.get_property(properties, self.I[2], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
        # POS: input(0)
        if len(self.I) > 3:
            features.append(self.get_property(properties, self.I[3], "pos"))
            features.append(self.get_property(properties, self.I[3], "morph"))
        else:
            features.append("NULL")
            features.append("NULL")
    
        if len(self.S)>0:
            top_pos = self.S[-1]
            leftmost = top_pos
            left_label = "NULL"
            rightmost = top_pos
            right_label = "NULL"
            dep_0 = "NULL"
            for (head, label, dep) in self.A:
                if dep==top_pos and dep_0!=None:
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
        if len(self.I)>0:
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


        if len(self.S)>0:
            top = self.S[-1]
            #word at the top of the stack
            top_word = self.get_property(properties, top, "word")
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
    def extract_train_features(self, sentence, properties):
        features = []
        modified_features = []
        self.S = []
        self.I = [i for i in range(0, len(sentence))]
        positions = [(self.I[i], sentence[i]) for i in range(0, len(sentence))]
        self.mappings = dict(positions)
        self.A = [] # the list of dependents on any given token
        i = 0
        while len(self.I)>0:

            state = self.get_action_features(properties)
            if len(self.S)==0:
                features.append(("S", 'NULL', state))
                self.S.append(self.I[0])
                self.I = self.I[1:]
            else:
                stack_top = self.S[-1]
                stack_head = self.get_property(properties, stack_top, "head")
                input_head = self.get_property(properties, self.I[0], "head")
                if stack_head == self.I[0]:
                    self.S.pop()
                    dep = self.get_property(properties, stack_top, "dep")
                    features.append(("LA", dep, state))
                    self.A.append((stack_head, dep, stack_top))
                elif input_head == stack_top:
                    dep = self.get_property(properties, self.I[0], "dep")
                    features.append(("RA", dep, state))
                    self.A.append((input_head, dep, self.I[0]))
                    self.S.append(self.I[0])
                    self.I = self.I[1:]

                else:
                    features.append(("S", 'NULL', state))
                    self.S.append(self.I[0])
                    self.I = self.I[1:]
        #final state
        state = self.get_action_features(properties)
        features.append(("END", 'NULL', state))


        for (state, dep, feat) in features:

            feat = [self.mappings[f] if f in self.mappings.keys() else f for f in feat]
            # print feat
            # print "%s\t%s\t%s" % (state, dep, "\t".join(feat))
            modified_features.append((state, dep, feat))
        self.A = [(self.mappings[h], l, self.mappings[d]) for (h,l,d) in self.A]
        # print "Dependencies: %s\n" % self.A

        return modified_features

    def extract_test_features(self, sentence, properties):
        # print "extracting"
        features = []
        modified_features = []
        self.S = []
        self.I = [i for i in range(0, len(sentence))]
        positions = [(self.I[i], sentence[i]) for i in range(0, len(sentence))]
        self.mappings = dict(positions)
        self.A = [] # the list of dependents on any given token
        i = 0
        while len(self.I)>0:
            state = self.get_action_features(properties)
            lex_state = [self.mappings[f] if f in self.mappings.keys() else f for f in state]
            fv = convert_instance_to_fv(lex_state)
            (action, dep) = self.predict(fv)
            features.append((action, dep, state))
            if action == actions["S"]:
                self.S.append(self.I[0])
                self.I = self.I[1:]
            elif action == actions["LA"]:
                stack_top = self.S[-1]
                self.S.pop()
                properties[stack_top][columns["head"]] = self.I[0]
                properties[stack_top][columns["head"]] = FV_MAPPINGS[LABEL][dep]
                self.A.append((self.I[0], dep, stack_top))
            elif action == actions["RA"]:
                stack_top = self.S[-1]
                properties[self.I[0]][columns["head"]] = stack_top
                properties[self.I[0]][columns["head"]] = FV_MAPPINGS[LABEL][dep]
                self.A.append((stack_top, dep, self.I[0]))
                self.S.append(self.I[0])
                self.I = self.I[1:]


        state = self.get_action_features(properties)
        features.append(("END", 'NULL', state))

        for (action, dep, state) in features:

            state = [self.mappings[f] if f in self.mappings.keys() else f for f in state]
            # print feat
            # print "%s\t%s\t%s" % (state, dep, "\t".join(feat))
            # print len(feat)
            modified_features.append((action, dep, state))
        modified_A = [(self.mappings[h], FV_MAPPINGS[LABEL][l], self.mappings[d]) for (h,l,d) in self.A]
        self.A = [(h, FV_MAPPINGS[LABEL][l], d) for (h,l,d) in self.A]
        # print "Dependencies: %s\n" % modified_A

        return self.A

    def predict(self, test_vectors):

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
        smallest = nsmallest(k, min_distances)
        # print smallest
        next_action = defaultdict(int)

        next_dep = defaultdict(int)
        for (dist, pos) in smallest:
            fv = training_fvs[pos]
            action_num = fv[0]
            dep_num = fv[1]
            next_action[action_num] += 1
            next_dep[dep_num] += 1
        # print next_action
        # print next_dep
        chosen_action = max(next_action.iteritems(), key=operator.itemgetter(1))[0]
        chosen_dep = max(next_dep.iteritems(), key=operator.itemgetter(1))[0]
        if chosen_action == actions["S"] and chosen_dep!=0:
            chosen_dep = 0
        elif chosen_action != actions["S"] and chosen_dep==0:
            chosen_dep = max([(key, value) for (key,value) in next_dep.iteritems() if value!=0], key=operator.itemgetter(1))[0]


        # print "%d\t%d" %(chosen_action,chosen_dep)
        return (chosen_action, chosen_dep)





    def reset(self):
        self.S = []
        self.I = []
        self.A = [] # the list of dependents on any given token
        self.mappings = {}

def add_fv_mappings(features):
    global FV_MAPPINGS
    for f in features:
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
        for f in sent_features:
            action_name = f[0]
            action_num = actions[action_name]
            dep = f[1]
            state = f[2]
            dep_num = FV_MAPPINGS[LABEL].index(dep)
            vectors = convert_instance_to_fv(state)
            line = [("%d:%s" % (f, v)) for (f,v) in vectors.iteritems()]
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
        # else:
        #     print "oops!"
        v = "".join(v)
        vectors[i] = v
    return vectors

def train(filepath, max=-1, start=1):
    global FV_MAPPINGS
    FV_MAPPINGS = defaultdict(lambda: ["NULL"])
    training_features = []
    parser = Parser()
    infile = open(filepath,'r')
    line = infile.readline()
    first = True
    properties = {}
    sentence = None
    num = 0

    while line:
        line = line.strip()
        line = line.lower()
        if line=="":

            # get features here
            parser.reset()
            num +=1
            if num>=start:
                sent_features = parser.extract_train_features(sentence, properties)
                add_fv_mappings(sent_features)
                training_features.append(sent_features)

            if num==max+start-1:
                break
            properties = {}
            first = True


        elif first:
            line = line.split()
            line = [w.strip() for w in line]
            sentence = line
            print "%d:\t%s" % (num, sentence)
            first = False
        else:
            line = line.split('\t')
            pos = int(line[columns["index"]])
            properties[pos] = line[columns["index"]+1:]


        line = infile.readline()

    convert_to_fvs(training_features, "../training.dat")
    infile.close()
def get_raw_accuracy(predictions, test_path, predicted_sentences):
    infile = open(test_path, 'r')
    line = infile.readline()
    first = True
    sentence = None
    dependencies = []
    num = 0
    total = 0.0
    correct = 0.0

    while line:
        line = line.strip()
        line = line.lower()
        if line=="":
            if sentence in predicted_sentences:
                for dep in dependencies:
                    total+=1
                    if dep in predictions[sentence]:
                        correct+=1
                    else:
                        print dep
                        print predictions[sentence]

            num +=1
            dependencies = []
            first = True


        elif first:
            sentence = line
            first = False
        else:
            line = line.split()
            dependencies.append((int(line[0]), line[1].lower(), int(line[2])))

        line = infile.readline()

    infile.close()

    print "Accuracy: %2.2f" % (correct/total)

def predict(filepath, max =-1, start=1):
    parser = Parser()
    infile = open(filepath,'r')
    line = infile.readline()
    first = True
    properties = defaultdict(list)
    tested_sentences = []
    sentence = None
    num = 0
    predictions = {}
    while line:
        line = line.strip()
        line = line.lower()
        if line=="":

            # get features here
            parser.reset()
            num +=1
            if num>=start:
                tested_sentences.append(" ".join(sentence))
                sent_pred = parser.extract_test_features(sentence, properties)
                predictions[tested_sentences[-1]] = sent_pred



            if num==max+start-1:
                break
            properties = {}
            first = True


        elif first:

            line = line.split()
            line = [w.strip() for w in line]
            sentence = line
            # print "%d:\t%s" % (num, sentence)
            first = False
        else:
            line = line.split('\t')
            pos = int(line[columns["index"]])
            properties[pos] = line[columns["index"]+1:]
            properties[pos].append('NULL')
            properties[pos].append('NULL')


        line = infile.readline()

    infile.close()
    print predictions
    get_raw_accuracy(predictions, '../real_test.txt', tested_sentences)


train('../welt-annotation-spatial.txt', 1, 4)
predict('../test_sentences.txt', max = 2)