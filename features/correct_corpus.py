# Written by: Anusha Balakrishnan
#Date: 12/2/14
import itertools
from features.data_tools import DataExtractor, get_property, DataParser
from features.nivre import Parser

def get_dependencies_from_properties(real_properties):
        all_deps = []
        for index in real_properties:
            head = get_property(real_properties, index, "head")
            label = get_property(real_properties, index, "dep")
            all_deps.append((head, label, index))
        return all_deps

def get_mismatches(predictions, test_dependencies):
        mismatches = 0

        for (head, label, dep) in test_dependencies:
            found = False
            for (pred_head, pred_label, pred_dep) in predictions:
                if head == pred_head and dep == pred_dep:
                    if label == pred_label:
                        found = True
            if not found:
                mismatches += 1

        return mismatches
def remove_large_sentences(infile, outfile):
    data_parser = DataParser()
    data_parser.load_data(infile, max_sentence_size=30)
    data_parser.write_data(outfile)
    print "%d sentences with < 30 tokens." % len(data_parser.all_data)
def remove_non_projective(infile, outfile, limit=-1):
    extractor = DataExtractor(lang=None, conll_format=True)
    data, morph_feats = extractor.get_sentences(infile)
    data_parser = DataParser()
    data_parser.load_data(infile)

    parser = Parser(morph_feats)
    num = 0
    fixed_data = []
    print infile
    for ((sentence, properties), lines) in itertools.izip(data, data_parser.all_data):
        real_deps = get_dependencies_from_properties(properties)
        try:
            state = parser.get_state_sequence(sentence, properties, gold_standard=real_deps)
        except KeyError:
            print properties
        dependencies = parser.A
        mismatches = get_mismatches(dependencies, real_deps)
        if mismatches<=1:
            fixed_data.append(lines)
            if len(fixed_data)==limit:
                break
        else:
            num += 1
            if num%500==0:
                print num
            # print sentence

    print "%d projective sentences." % len(fixed_data)
    data_parser.write_data(outfile, fixed_data)
    print("Non-projective sentences: %d" % num)

# remove_large_sentences('../treebanks/dutch_alpino_train.conll', '../treebanks/dutch_short_sentences.conll')
# remove_non_projective('../treebanks/dutch_short_sentences.conll', '../treebanks/dutch_train.conll', limit=2000)
# remove_large_sentences('../treebanks/german_conll2009_train.conll', '../treebanks/german_short_sentences.conll')
# remove_non_projective('../treebanks/german_short_sentences.conll', '../treebanks/german_train.conll')
# remove_large_sentences('../treebanks/portuguese_bosque_train.conll', '../treebanks/portugese_short_sentences.conll')
# remove_non_projective('../treebanks/portugese_short_sentences.conll', '../treebanks/portugese_train.conll')
# remove_large_sentences('../treebanks/swedish_talbanken05_train.conll', '../treebanks/swedish_short_sentences.conll')
# remove_non_projective('../treebanks/swedish_short_sentences.conll', '../treebanks/swedish_train.conll')

# remove_large_sentences('../treebanks/german_conll2009_train.conll', '../treebanks/german_short_sentences.conll')
# remove_non_projective('../treebanks/german_short_sentences.conll', '../treebanks/german_train.conll')
#
remove_large_sentences('../treebanks/slovene_treebank_train.conll', '../treebanks/slovene_short_sentences.conll')
remove_non_projective('../treebanks/slovene_short_sentences.conll', '../treebanks/slovene_train.conll')
#
# remove_non_projective('../welt-data/welt-train.txt', '../welt-data/welt-train-corrected.txt')