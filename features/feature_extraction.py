#Written by: Anusha Balakrishnan
#Date: 11/21/14
from collections import defaultdict
import pickle


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
                    if isinstance(state[i], list):
                        for val in state[i]:
                            self.FV_MAPPINGS[i].append(val)
                    else:
                        self.FV_MAPPINGS[i].append(state[i])

    def convert_instance_to_fv(self, state):
        vectors = defaultdict(str)
        for i in range(0, len(state)):
            v = ["0"] * len(self.FV_MAPPINGS[i])
            if isinstance(state[i], list):
                for val in state[i]:
                    if val in self.FV_MAPPINGS[i]:
                        v[self.FV_MAPPINGS[i].index(val)] = "1"
            else:
                if state[i] in self.FV_MAPPINGS[i]:
                    v[self.FV_MAPPINGS[i].index(state[i])] = "1"
            v = "".join(v)
            vectors[i] = v
        return vectors

    def convert_to_fvs(self, all_features):
        training_fvs = []
        for sent_features in all_features:
            # print sent_features
            for (action, dep, state) in sent_features:
                action_num = action.value
                dep_num = self.FV_MAPPINGS[self.LABEL].index(dep)
                vectors = self.convert_instance_to_fv(state)
                training_fvs.append((action_num, dep_num, vectors))
                # print "\t", vectors

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
            feature_val = 0
            if '1' in f:
                feature_val = f.index('1') + 1
            new_val.append(feature_val)
        return new_val

    def get_index(self, feature_index, feature_val):
        if feature_index not in self.FV_MAPPINGS.keys():
            return -1
        if feature_val not in self.FV_MAPPINGS[feature_index]:
            return -1
        return self.FV_MAPPINGS[feature_index].index(feature_val)