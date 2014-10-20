# Written by: Anusha Balakrishnan
#Date: 10/6/14
columns = {"index":0, "word":1, "stem":2, "morph":3, "pos":4, "head":5, "dep":6}
LABEL = "LABEL"
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
        # POS:stack(1)
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
            features.append(self.get_property(properties, top, "word"))
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
    def extract_features(self, sentence, properties):
        features = []
        self.S = []
        self.I = [i for i in range(0, len(sentence))]
        positions = [(self.I[i], sentence[i]) for i in range(0, len(sentence))]
        self.mappings = dict(positions)
        self.A = [] # the list of dependents on any given token
        i = 0
        while len(self.I)>0:
            state = self.get_action_features(properties)
            if len(self.S)==0:
                features.append(("S", state))
                self.S.append(self.I[0])
                self.I = self.I[1:]
            else:
                stack_top = self.S[-1]
                stack_head = self.get_property(properties, stack_top, "head")
                input_head = self.get_property(properties, self.I[0], "head")
                if stack_head == self.I[0]:
                    features.append(("LA", state))
                    self.S.pop()
                    dep = self.get_property(properties, stack_top, "dep")
                    self.A.append((stack_head, dep, stack_top))
                elif input_head == stack_top:
                    features.append(("RA", state))
                    dep = self.get_property(properties, self.I[0], "dep")
                    self.A.append((input_head, dep, self.I[0]))
                    self.I = self.I[1:]
                else:
                    features.append(("S", state))
                    self.S.append(self.I[0])
                    self.I = self.I[1:]
        #final state
        state = self.get_action_features(properties)
        features.append(("END", state))
        for (state, feat) in features:
            feat = [self.mappings[f] if f in self.mappings.keys() else f for f in feat]
            # print feat
            print "%s\t%s" % (state, "\t".join(feat))
        self.A = [(self.mappings[h], l, self.mappings[d]) for (h,l,d) in self.A]
        print "Dependencies: %s\n" % self.A

    def reset(self):
        self.S = []
        self.I = []
        self.A = [] # the list of dependents on any given token
        self.mappings = {}

def convert_to_features(filepath):
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
            parser.extract_features(sentence, properties)
            # break
            properties = {}
            first = True
            num +=1

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
    parser.reset()
    parser.extract_features(sentence, properties)
    infile.close()

convert_to_features('../welt-annotation-spatial.txt')
