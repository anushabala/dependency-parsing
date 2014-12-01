#Written by: Anusha Balakrishnan
#Date: 11/21/14
from random import randint
from nltk.stem import snowball


columns = {"index": 0, "word": 1, "stem": 2, "morph": 3, "pos": 4, "head": 5, "dep": 6}
conll_columns = {"index": 0, "word": 1, "stem": 2, "cpos": 3, "fpos": 4, "head": 5, "feats":6, "deprel": 7, "phead": 8,
                 "pdeprel": 9}
class DataParser:
    def __init__(self):
        self.all_data = []
        self.training_data = []
        self.test_data = []
        self.selected = 0
    def write_fvs(self, fvs, filepath):
        out_file = open(filepath, 'w')
        for (action, dep, vectors) in fvs:
            vecs = [("%d:%s" % (f, v)) for (f, v) in vectors.iteritems()]
            vecs = "\t".join(vecs)
            line = "%s\t%s\t%s\n" % (str(action), str(dep), vecs)
            out_file.write(line)
        out_file.close()

    def load_data(self, filepath):
        # print("[Testing]")
        sentences = set()
        current_sentence = None
        first = True

        infile = open(filepath, 'r')
        line = infile.readline()

        properties = []
        while line:
            line = line.strip()
            line = line.lower()
            if line == "":
                if current_sentence not in sentences:
                    self.all_data.append(properties)
                    sentences.add(current_sentence)
                else:
                    # print "Sentence %s repeated" % current_sentence
                    pass
                properties = []
                first = True
            else:
                line = line.strip()
                if first:
                    current_sentence = line
                    first = False
                properties.append(line)
            line = infile.readline()

        infile.close()
        return len(self.all_data)

    def choose_training_data(self, train_file, train_num, validation_data=None, is_ratio=True):

        while self.selected<train_num and self.selected<len(self.all_data):
            pos = randint(self.selected, len(self.all_data)-1)
            self.training_data.append(self.all_data[pos])
            self.all_data[pos], self.all_data[self.selected] = self.all_data[self.selected], self.all_data[pos]
            self.selected += 1

        if validation_data!=None:
            while self.selected<len(self.all_data):
                self.test_data.append(self.all_data[self.selected])
                self.selected+=1

        train_file = open(train_file, 'w')
        for sentence in self.training_data:
            for line in sentence:
                train_file.write(line+"\n")

            train_file.write("\n")
        train_file.close()

        if validation_data!=None:
            validation_data = open(validation_data, 'w')
            for sentence in self.test_data:
                for line in sentence:
                    validation_data.write(line+"\n")
                validation_data.write("\n")
            validation_data.close()

        # print self.training_data
        return (len(self.training_data), len(self.test_data))

    def initial_split(self, train_file, test_file):
        self.reset_splits()
        train_file = open(train_file, 'w')
        test_file = open(test_file, 'w')
        ratio = 0.9
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

    def fold_split(self, train_file, test_file, fold, total_folds=10):
        self.reset_splits()
        train_file = open(train_file, 'w')
        test_file = open(test_file, 'w')
        ratio = total_folds/100.0
        test_num = int(ratio * len(self.all_data))
        fold_start = int((fold-1)*(len(self.all_data)*ratio))
        fold_end = fold_start+test_num
        test_data = self.all_data[fold_start:fold_end]
        train_data = self.all_data[0:fold_start]
        train_data.extend(self.all_data[fold_end:])

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

    def reset_splits(self):
        self.training_data = []
        self.test_data = []
        self.selected = 0

def get_property(properties, pos, propName):
    relevant_props = properties[pos]
    if propName == "head":
        return int(relevant_props[columns[propName] - 1])
    return relevant_props[columns[propName] - 1]


def set_property(properties, pos, propName, new_value):
    relevant_props = properties[pos]
    relevant_props[columns[propName] - 1] = new_value

def get_sentences(filepath, print_status=False):

    # print("[Training]")
    infile = open(filepath, 'r')
    line = infile.readline()
    first = True
    properties = {}
    sentence = None
    data = []
    num = 0
    morph_feats = set()

    while line:
        line = line.strip()
        line = line.lower()
        if line == "":
            num +=1
            data.append((sentence, properties))

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
            line = ["NULL" if w=='' or w=='_' else w for w in line]
            line[columns["morph"]] = line[columns["morph"]].split('|')
            for feat in line[columns["morph"]]:
                if feat!='NULL':
                    morph_feats.add(feat)
            index_pos = columns["index"]
            pos = int(line[index_pos])
            properties[pos] = line[index_pos + 1:]
        line = infile.readline()
    infile.close()

    new_data = []
    for (sentence, properties) in data:

        for key in properties:
            current_morph = get_property(properties, key, "morph")
            new_morph = []
            for f in morph_feats:
                if f in current_morph:
                    new_morph.append(f)
                else:
                    new_morph.append("_")
            set_property(properties, key, "morph", new_morph)


        new_data.append((sentence, properties))

    data = new_data
    # print data
    # print morph_feats
    return data, morph_feats

#todo: "_" is also NULL including ""

def add_word_roots(filepath, out_file, lang="english"):
    infile = open(filepath, 'r')
    out_file = open(out_file, 'w')
    stemmer = snowball.GermanStemmer()
    stem_pos = 2
    for line in infile.readlines():
        line = line.strip()
        if len(line)!=0:
            line = line.split('\t')
            form = line[1]
            # todo: make unicode
            line[stem_pos] = stemmer.stem(form)
            line = "\t".join(line)
            out_file.write(line+"\n")
        else:
            out_file.write("\n")

    infile.close()
    out_file.close()

# add_word_roots('../universal_dependencies/de/de-universal-train.conll', '../universal_dependencies/de/de-fixed.conll')
get_sentences('../test_data.txt')