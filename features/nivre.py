#Written by: Anusha Balakrishnan
#Date: 11/21/14
from features.convert_dependencies import get_property, actions, set_property


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