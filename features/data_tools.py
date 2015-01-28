#Written by: Anusha Balakrishnan
#Date: 11/21/14
import copy
from random import randint
import itertools
from nltk.stem import snowball


columns = {"index": 0, "word": 1, "stem": 2, "morph": 3, "pos": 4, "head": 5, "dep": 6}
# conll_columns = {"index": 0, "word": 1, "stem": 2, "cpos": 3, "fpos": 4, "feats": 5, "head": 6, "deprel": 7, "phead": 8,
#                  "pdeprel": 9}
conll_columns = {"index": 0, "word": 1, "stem": 2, "plemma": 3, "cpos": 4, "ppos": 5, "feats": 6, "pfeat": 7, "head": 8, "phead": 9,
                 "deprel": 10, "pdeprel": 11}

class DataParser:
    def __init__(self):
        self.all_data = []
        self.training_data = []
        self.test_data = []
        self.selected = 0
    def write_fvs(self, fvs, filepath):
        out_file = open(filepath, 'w')
        for (action, dep, vectors) in fvs:
            vecs = [("%d:%s" % (f, v)) for f, v in itertools.izip(range(0, len(vectors)), vectors)]
            vecs = "\t".join(vecs)
            line = "%s\t%s\t%s\n" % (str(action), str(dep), vecs)
            out_file.write(line)
        out_file.close()

    def write_data(self, filepath, data=None):
        if data==None:
            data = self.all_data
        out_file = open(filepath, 'w')
        for d in data:
            for line in d:
                out_file.write(line+"\n")
            out_file.write("\n")

        out_file.close()
    def load_data(self, filepath, limit=-1, max_sentence_size=-1):
        # print("[Testing]")
        # sentences = set()
        # current_sentence = None
        # first = True

        infile = open(filepath, 'r')
        line = infile.readline()
        num = 0
        properties = []
        punct = 0
        while line:
            line = line.strip()
            line = line.lower()
            if line == "":
                # if current_sentence not in sentences:
                if max_sentence_size>0 and len(properties)-punct<=max_sentence_size:
                    self.all_data.append(properties)
                    num += 1
                elif max_sentence_size<0:
                    self.all_data.append(properties)
                    num += 1
                #     sentences.add(current_sentence)
                # else:
                #     # print "Sentence %s repeated" % current_sentence
                #     pass

                if num==limit:
                    break
                properties = []
                punct = 0
                # first = True
            else:
                line = line.strip()
                parts = line.split('\t')
                label = parts[-3]
                pos = parts[4]
                if label=='punc' or label == 'punct' or label == 'pnct' or '$' in pos or pos=='punc':
                    punct += 1
                # if first:
                #     current_sentence = line
                #     first = False
                properties.append(line)
            line = infile.readline()

        infile.close()
        return len(self.all_data)

    def choose_training_data_random(self, train_file, train_num, is_ratio=True):

        while self.selected<train_num and self.selected<len(self.all_data):
            pos = randint(self.selected, len(self.all_data)-1)
            self.training_data.append(self.all_data[pos])
            self.all_data[pos], self.all_data[self.selected] = self.all_data[self.selected], self.all_data[pos]
            self.selected += 1

        train_file = open(train_file, 'w')
        for sentence in self.training_data:
            for line in sentence:
                train_file.write(line+"\n")

            train_file.write("\n")
        train_file.close()

        # print self.training_data
        return len(self.training_data)

    def choose_training_data(self, train_file, train_num, is_ratio=True):
        self.training_data = self.all_data[:train_num]
        train_file = open(train_file, 'w')
        for sentence in self.training_data:
            for line in sentence:
                train_file.write(line+"\n")

            train_file.write("\n")
        train_file.close()

        return len(self.training_data)

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

class DataExtractor:
    def __init__(self, lang, conll_format=True, fine_tags=False):
        self.conll_format = conll_format
        self.fine_tags = fine_tags
        self.lang = lang
        if lang=="german":
            global conll_columns
            self.stemmer = snowball.GermanStemmer()
        elif lang=="english":
            self.stemmer = snowball.EnglishStemmer()
        elif lang=="danish":
            self.stemmer = snowball.DanishStemmer()
        elif lang=="dutch":
            self.stemmer = snowball.DutchStemmer()
        elif lang=="swedish":
            self.stemmer = snowball.SwedishStemmer()
        elif lang=="portugese":
            self.stemmer = snowball.PortugueseStemmer()
        else:
            self.stemmer = None

    def get_sentences(self, filepath, print_status=False, morph_feats=None):
        # print("[Training]")
        infile = open(filepath, 'r')
        line = infile.readline()
        properties = {}
        sentence = []
        data = []
        num = 0
        find_morphs = False
        if morph_feats==None:
            find_morphs = True
            morph_feats = set()

        while line:

            skip_sentence = False
            line = line.strip()
            line = line.lower()
            if line == "":
                num +=1
                properties = self.clean_up(properties)
                sentence = self.get_sentence(properties)
                if properties!={}:
                    data.append((sentence, properties))

                if print_status:
                    print "%d:\t%s" % (num, sentence)
                properties = {}
                sentence = []


            else:
                orig = line
                line = line.split('\t')
                # print line
                line = [w.strip() for w in line]
                line = ["NULL" if w=='' or w=='_' else w for w in line]
                if self.conll_format:
                    line = self.reformat_conll(line, fine=self.fine_tags)
                if line==[]:
                    skip_sentence = True

                try:
                    label = line[columns["dep"]]
                    part_of_speech = line[columns["pos"]]
                    line[columns["head"]] = int(line[columns["head"]])

                    if line[columns["stem"]]=="_":
                        line[columns["stem"]] = self.get_word_root(line[columns["word"]])


                    line[columns["morph"]] = line[columns["morph"]].split('|')

                    line[columns["morph"]] = [w.split("=")[1] if "=" in w else w for w in line[columns["morph"]]]

                    if find_morphs:
                        for feat in line[columns["morph"]]:
                            if feat!='NULL':
                                morph_feats.add(feat)
                    index_pos = columns["index"]

                    pos = int(line[index_pos])
                    if pos>20:
                        skip_sentence = True
                    if self.conll_format:
                        pos -= 1
                        line[columns["head"]] = int(line[columns["head"]]) - 1
                    properties[pos] = line[index_pos + 1:]
                except ValueError as ve:
                    skip_sentence = True
                    print "error on ", orig



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
        # print len(data)
        return data, morph_feats

    def clean_up(self, properties):
        offset = 0
        heads = set()
        new_properties = {}

        for key in properties:
            head = get_property(properties, key, "head")
            heads.add(head)

        new_positions = {}
        for key in properties.keys():
            pos = get_property(properties, key, "pos")
            dep = get_property(properties, key, "dep")
            if dep == "punc" or dep == "punct" or dep == "pnct" or pos == "punc" or '$' in pos:
                if key not in heads:
                    offset+= 1
                    continue
            new_key = key - offset
            new_positions[key] = new_key
            new_properties[new_key] = copy.deepcopy(properties[key])

        for key in new_properties.keys():
            head = get_property(new_properties, key, "head")
            if head!=-1:
                new_key = new_positions[head]
            else:
                new_key = -1
            set_property(new_properties, key, "head", new_key)

        return new_properties

    def get_sentence(self, properties):
        sentence = []
        for key in properties.keys():
            sentence.append(get_property(properties, key, "word"))
        return sentence


    def reformat_conll(self, line, fine=False):
        try:
            new_line = []
            if '_' in line[conll_columns["index"]]:
                line[conll_columns["index"]] = line[conll_columns["index"]].split('_')[1]
            new_line.append(line[conll_columns["index"]])

            new_line.append(line[conll_columns["word"]])
            new_line.append(line[conll_columns["stem"]])
            if line[conll_columns["feats"]]=='_':
                new_line.append(line[conll_columns["fpos"]])
            else:
                new_line.append(line[conll_columns["feats"]])

            if fine:
                new_line.append(line[conll_columns["fpos"]])
            else:
                new_line.append(line[conll_columns["cpos"]])
            new_line.append(line[conll_columns["head"]])
            new_line.append(line[conll_columns["deprel"]])
            return new_line
        except IndexError:
            # print line
            return []



    def get_word_root(self, word):
        word = word.decode('utf-8')
        if self.stemmer == None:
            return word

        stem = self.stemmer.stem(word)
        return stem
def get_property(properties, pos, propName):
    relevant_props = properties[pos]
    if propName == "head":
        return int(relevant_props[columns[propName] - 1])
    return relevant_props[columns[propName] - 1]

def set_property(properties, pos, propName, new_value):
    relevant_props = properties[pos]
    relevant_props[columns[propName] - 1] = new_value

def find_number_of_words(data):
    num = 0
    for sentence in data:
        for line in sentence:
            line = line.split('\t')
            if line[-3] != 'punct' and line[-3] != 'pnct' and line[-3] != 'punc':
                num += 1

    print "Number of words: %d" % num

def find_average_length(data):
    total = 0
    for sentence in data:
        for line in sentence:
            line = line.split('\t')
            pos = line[4]
            dep = line[-2]
            if  '$' not in pos:
                total += 1
    total = total/float(len(data))
    print "Average length: %2.1f" % total

def find_total_morph_features(filepath, lang):
    extractor = DataExtractor(lang=lang, conll_format=True)
    data, morph_feats = extractor.get_sentences(filepath)
    print morph_feats
    print len(morph_feats)

def find_total_deps(data):
    deps = set()
    for sentence in data:
        for line in sentence:
            line = line.split('\t')
            dep = line[-1]
            deps.add(dep)

    print len(deps)
    print deps

# add_word_roots('../universal_dependencies/de/de-universal-train.conll', '../universal_dependencies/de/de-fixed.conll')
# parser = DataParser()
# parser.load_data('../treebanks/temp_train.conll', max_sentence_size=30)
# parser.write_data('../treebanks/temp.txt', parser.all_data)
# parser = DataExtractor(lang="en", conll_format=False)
# data, morph_features = parser.get_sentences('../welt-data/welt-train.txt')
# for (sentence, properties) in data:
#     for key in properties.keys():
#         print key,"\t",properties[key]
#     print "\n"
# print len(data)

# parser = DataParser()
# parser.load_data('../treebanks/german_train.conll', limit=1200)
# find_average_length(parser.all_data)
# find_number_of_words(parser.all_data)

# find_total_morph_features('../treebanks/german_train.conll', "german")

# extractor = DataExtractor(lang=None, conll_format=True)
# data, morph_feats = extractor.get_sentences("../treebanks/german_train.conll")
# for (sentence, properties) in data:
#     for key in properties.keys():
#         print key,"\t",properties[key]
#     print "\n"
# print len(data)
#
# parser = DataParser()
# parser.load_data('../welt-data/welt-train.txt', limit=1200)
# find_total_deps(parser.all_data)
# find_average_length(parser.all_data)