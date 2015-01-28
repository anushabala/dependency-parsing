#Written by: Anusha Balakrishnan
#Date: 11/21/14
from features.data_tools import get_property, set_property
from enum import Enum

class ParserActions(Enum):
    S = 1
    LA = 2
    RA = 3
    REDUCE = 4
    END = 5

class Parser:

    def __init__(self, morph_features, k=1):
        self.S = []
        self.I = []
        self.A = []  # the list of dependents on any given token
        self.mappings = {}
        self.morph_features = morph_features



    def get_current_state(self, properties):
        features = []
        # Feature 1: POS:stack(1), morph:stack(1)
        if len(self.S) > 1:
            features.append(get_property(properties, self.S[-2], "pos"))
        else:
            features.append("NULL")

        # POS:stack(0)
        if len(self.S) > 0:
            features.append(get_property(properties, self.S[-1], "pos"))
        else:
            features.append("NULL")

        # POS: input(0)
        if len(self.I) > 0:
            features.append(get_property(properties, self.I[0], "pos"))
        else:
            features.append("NULL")
        # POS: input(1)
        if len(self.I) > 1:
            features.append(get_property(properties, self.I[1], "pos"))
        else:
            features.append("NULL")
        # POS: input(2)
        if len(self.I) > 2:
            features.append(get_property(properties, self.I[2], "pos"))
        else:
            features.append("NULL")
            # features.append("NULL")
        # POS: input(3)
        if len(self.I) > 3:
            features.append(get_property(properties, self.I[3], "pos"))
        else:
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
            #word at the top of the stack and its morph
            top_word = get_property(properties, top, "stem")
            features.append(top_word)
            morphology = get_property(properties, top, "morph")
            for f in self.morph_features:
                if f in morphology:
                    features.append(f)
                else:
                    features.append("_")
            top_head = "NULL"
            for (head, label, dep) in self.A:
                if dep == top:
                    top_head = head
                    break
            #head of word at stack top
            features.append(top_head)

        else:
            features.append("NULL")
            for i in range(0, len(self.morph_features)):
                features.append("NULL")
            features.append("NULL")

        #current input word
        if len(self.I) > 0:
            features.append(get_property(properties, self.I[0], "stem"))
            morphology = get_property(properties, self.I[0], "morph")
            for f in self.morph_features:
                if f in morphology:
                    features.append(f)
                else:
                    features.append("_")
        else:
            features.append("NULL")
            for i in range(0, len(self.morph_features)):
                features.append("NULL")
        #next input word
        if len(self.I) > 1:
            features.append(get_property(properties, self.I[1], "stem"))
            morphology = get_property(properties, self.I[1], "morph")
            for f in self.morph_features:
                if f in morphology:
                    features.append(f)
                else:
                    features.append("_")
        else:
            features.append("NULL")
            for i in range(0, len(self.morph_features)):
                features.append("NULL")

        return features

    def get_state_sequence(self, sentence, properties, gold_standard=None):
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
                features.append((ParserActions.S, 'NULL', state))
                self.shift()

            else:
                stack_top = self.S[-1]
                stack_head = get_property(properties, stack_top, "head")
                input_head = get_property(properties, self.I[0], "head")
                if stack_head == self.I[0] and not self.has_head(stack_top):
                    dep = get_property(properties, stack_top, "dep")
                    features.append((ParserActions.LA, dep, state))
                    self.left_arc(stack_head, dep, stack_top)


                elif input_head == stack_top and not self.has_head(self.I[0]):
                    dep = get_property(properties, self.I[0], "dep")
                    features.append((ParserActions.RA, dep, state))
                    self.right_arc(input_head, dep)



                else:
                    if gold_standard!=None:
                        if self.found_all_dependents(stack_top, gold_standard) and self.has_head(stack_top):
                            features.append((ParserActions.REDUCE, 'NULL', state))
                            self.reduce()
                        else:
                            features.append((ParserActions.S, 'NULL', state))
                            self.shift()
                    else:
                        if self.has_head(stack_top):
                            features.append((ParserActions.REDUCE, 'NULL', state))
                            self.reduce()
                        else:
                            features.append((ParserActions.S, 'NULL', state))
                            self.shift()

        #final state
        state = self.get_current_state(properties)
        features.append((ParserActions.END, 'NULL', state))

        for (state, dep, feat) in features:
            feat = [self.mappings[f] if f in self.mappings.keys() else f for f in feat]
            modified_features.append((state, dep, feat))
        self.A = [(h, l, d) for (h, l, d) in self.A]
        return modified_features

    def found_all_dependents(self, token, gold_standard):
        total_deps = 0
        for (head, label, dep) in gold_standard:
            if token==head:
                total_deps += 1
        found_deps = 0
        for (head, label, dep) in self.A:
            if token==head:
                found_deps += 1
        return total_deps==found_deps
    def has_head(self, token):
        for (head, label, dep) in self.A:
            if dep==token:
                return True
        return False
    def left_arc(self, head, label, dep):
        self.S.pop()
        self.A.append((head, label, dep))
    def right_arc(self, head, label):
        self.A.append((head, label, self.I[0]))

        # self.S.pop()
        # self.I[0] = head

        self.S.append(self.I[0])
        self.I = self.I[1:]

    def shift(self):
        self.S.append(self.I[0])
        self.I = self.I[1:]

    def reduce(self):
        self.S.pop()
    def predict_actions(self, sentence, properties, classifier):
        # print "extracting"
        features = []
        modified_features = []
        self.S = []
        self.I = [i for i in range(0, len(sentence))]
        positions = [(self.I[i], sentence[i]) for i in range(0, len(sentence))]
        self.mappings = dict(positions)
        self.A = []  # the list of dependents on any given token
        alternate_deps = []
        i = 0
        while len(self.I) > 0:
            state = self.get_current_state(properties)
            lex_state = [self.mappings[f] if f in self.mappings.keys() else f for f in state]
            (action, dep) = classifier.get_next_action(lex_state)
            # todo: extra; remove these 2 lines for normal execution
            # alt_dep = classifier.get_extra_dep(lex_state)


            # print "Token: %s\tAction:%s\tdep:%s" % (str(self.I[0]), action, dep)
            # print "predicted action %s dep %s" % (str(action), str(dep))
            if (action == ParserActions.LA or action==ParserActions.RA or action == ParserActions.REDUCE) and len(self.S) == 0:
                action = ParserActions.S
            features.append((action, dep, state))
            if action == ParserActions.S:
                self.shift()

            elif action == ParserActions.REDUCE:
                self.reduce()

            elif action == ParserActions.LA:
                stack_top = self.S[-1]
                set_property(properties, stack_top, "head", self.I[0])
                set_property(properties, stack_top, "dep", dep)
                self.left_arc(self.I[0], dep, stack_top)
                # todo: etra
                # alternate_deps.append(alt_dep)

            elif action == ParserActions.RA:
                stack_top = self.S[-1]
                set_property(properties, self.I[0], "head", stack_top)
                set_property(properties, self.I[0], "dep", dep)
                self.right_arc(stack_top, dep)
                # todo: extra
                # alternate_deps.append(alt_dep)

            elif action == ParserActions.END:
                break

        state = self.get_current_state(properties)
        features.append((ParserActions.END, 'NULL', state))

        for (action, dep, state) in features:
            state = [self.mappings[f] if f in self.mappings.keys() else f for f in state]
            modified_features.append((action, dep, state))

        #todo: extra
        # new_A = []
        # i = 0
        # for (head, label, dep) in self.A:
        #     new_A.append((head, alternate_deps[i], dep))
        #     i+=1
        # self.A = new_A
        self.find_root(sentence)

        #todo: extra

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