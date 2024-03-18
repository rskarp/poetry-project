import json
import sys
from random import sample
import spacy
from nltk.corpus import wordnet
from nltk.tokenize import SyllableTokenizer
sys.path.insert(0, 'syllabifier')
from syllabifier import cmuparser3
from syllabifier.syllable3 import generate_syllables
from lemminflect import getInflection
from SoundsLike.SoundsLike import Search

class syllabify:
    def __init__(self, use_cmu = True):
        if use_cmu:
            self.syllabifier = cmuparser3.CMUDictionary()
        else:
            self.syllabifier = SyllableTokenizer()
        self.use_cmu = use_cmu
        
    def syllabify(self, term):
        if self.use_cmu:
            phoneme_str = self.syllabifier.get_first(term)
            if phoneme_str:
                return [str(x) for x in generate_syllables(phoneme_str)]
        else:
            return self.syllabifier.tokenize(term)

class poem_replacer:
    def __init__(self, lexicon_file, use_cmu = True, use_pos = True, use_anagrams = False, use_rhyme = True, use_syllables = True, union_or_intersection = 'union', min_wn_distance = 0, max_wn_distance = -1):
        self.nlp = spacy.load("en_core_web_md")
        with open(lexicon_file) as f:
            self.lexicon = json.load(f)
        self.syllabifier = syllabify(use_cmu)
        self.pos_mappings =  {'NOUN': wordnet.NOUN, 'VERB': wordnet.VERB, 'ADJ': wordnet.ADJ, 'ADV': wordnet.ADV}
        self.use_cmu = use_cmu
        self.use_pos = use_pos
        self.use_anagrams = use_anagrams
        self.use_rhyme = use_rhyme
        self.use_syllables = use_syllables
        self.union_or_intersection = union_or_intersection
        self.min_wn_distance = min_wn_distance
        self.max_wn_distance = max_wn_distance

    def get_candidates_by_pos(self, term, pos):
        if pos in self.lexicon['by_pos']:
            possibles = self.lexicon['by_pos'][pos]
            if possibles:
                return [x for x in possibles if x != term]
        return []
    
    def get_candidates_by_syllables(self, term):
        syllables = self.syllabifier.syllabify(term)
        if str(len(syllables)) in self.lexicon['by_syllables']:
            possibles = self.lexicon['by_syllables'][str(len(syllables))]
            if possibles:
                return [x for x in possibles if x != term]
        return []
        
    def get_candidates_by_anagrams(self, term):
        if ''.join(sorted(term)) in self.lexicon['by_anagrams']:
            possibles = self.lexicon['by_anagrams'][''.join(sorted(term))]
            if possibles:
                return [x for x in possibles if x != term]
        return []
    
    def get_candidates_by_rhymes(self, term):
        possibles = []
        if term in self.lexicon['by_rhymes']:
            possibles = self.lexicon['by_rhymes'][term]
        syllables = self.syllabifier.syllabify(term)
        if len(syllables) > 0 and syllables[-1] in self.lexicon['by_last_syllables']:
            possibles = list(set(possibles).union(self.lexicon['by_last_syllables'][syllables[-1]]))
        if possibles:
            return [x for x in possibles if x != term]
        return []
        
    def wn_distance(self, synset, possible, pos):
        possible_synsets = wordnet.synsets(possible, self.pos_mappings[pos])
        if len(possible_synsets) > 0:
            dist = synset.wup_similarity(possible_synsets[0], simulate_root=False)
            if dist is not None:
                return dist
        return float('inf')

    def get_candidates(self, token, end_of_line = False):
        pos_possibles = []
        syllables_possibles = []
        anagrams_possibles = []
        rhyme_possibles = []
        if self.use_pos:
            pos_possibles = self.get_candidates_by_pos(token.text, token.pos_)
        if self.use_syllables:
            syllables_possibles = self.get_candidates_by_syllables(token.text)
        if self.use_anagrams:
            anagrams_possibles = self.get_candidates_by_anagrams(token.text)
        if self.use_rhyme and end_of_line:
            rhyme_possibles = self.get_candidates_by_rhymes(token.text)
        if self.union_or_intersection == 'union':
            possibles = list(set(pos_possibles).union(syllables_possibles, anagrams_possibles, rhyme_possibles))
        else:
            possibles_all = [x for x in [pos_possibles, syllables_possibles, anagrams_possibles, rhyme_possibles] if len(x) > 0]
            if len(possibles_all) > 0:
                possibles = possibles_all[0]
                for possible in possibles_all[1:]:
                    possibles = list(set(possibles).intersection(possible))
            else:
                possibles = []
        # if min_wn_distance or max_wn_distance > -1, then we reduce candidates to those within min and max Wordnet distance of the token
        # there are multiple Wordnet distance algorithms; we just use wup for now; see https://www.nltk.org/howto/wordnet.html
        if self.min_wn_distance > 0:
            t = wordnet.synsets(token.text, self.pos_mappings[token.pos_])
            if len(t) > 0:
                if self.max_wn_distance > -1:
                    possibles = [possible for possible in possibles if len(wordnet.synsets(possible, self.pos_mappings[token.pos_])) > 0 and self.wn_distance(t[0], possible, token.pos_) in range(self.min_wn_distance, self.max_wn_distance)]
                else:
                    possibles = [possible for possible in possibles if len(wordnet.synsets(possible, self.pos_mappings[token.pos_])) > 0 and self.wn_distance(t[0], possible, token.pos_) > self.min_wn_distance]
        return possibles                                 
        
    def get_tokens(self, text):
        # make a spacy document from the input text
        doc = self.nlp(text)
        # get all the tokens
        # see https://spacy.io/api/token
        # see https://universaldependencies.org/u/pos/index.html
        tokens = [token for token in doc]
        content_tokens = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        return tokens, content_tokens               
    
    def process(self, text, percent_to_replace):                                 
        tokens, content_tokens = self.get_tokens(text)
        number_to_replace = int(len(content_tokens) * (percent_to_replace / 100))
        # choose number_to_replace tokens to replace
        to_replace = sample(content_tokens, number_to_replace)
        # this contains the output words
        words = []
        # for each token
        for i, token in enumerate(tokens):
            # if the token is to be replaced
            if token in to_replace:
                # figure out if it is at the end of the poem or the end of a line; you can use this if you want to only match rhyme at end of line
                end_of_line = (i == len(tokens)-1 or tokens[i+1].pos_ == 'SPACE')
                # get possible replacement terms
                possibles = self.get_candidates(token, end_of_line = end_of_line)
                # choose one of them at random and replace!
                if len(possibles) > 0:
                    chosen = sample(possibles, 1)
                    words.append("#REPLACE_TOKEN" + chosen[0] + token.whitespace_)
                else:
                    words.append(token.text + token.whitespace_)
            else:
                words.append(token.text + token.whitespace_)
        return number_to_replace, [x.text for x in to_replace], words

def main():
    text = '''You who in scattered rhymes listen to the sound 
    Of those sighs with which I fed the heart
    During that first youthful mistake of mine
    When I was in part a different man than I am now
    I hope I can find forgiveness and pithy 
    For the diverse style in which I cry and reason,
    Between useless hope and useless pain
    From those who understand love out of experience.
    Now I can see clearly how I have been a joke
    To all the people, for a long time, so much so that when I think about it,I often feel ashamed of myself; 
    Shame is the result of my ramblings,
    Along with regret, and the clear realization
    That what the world likes is but a brief dream'''

    # PR = poem_replacer('lexicon.json')
    # _, to_replace, words = PR.process(text, 10)
    # print('plain:', ', '.join(to_replace), ''.join(words))

    # PR = poem_replacer('lexicon.json', use_pos = True, use_anagrams = False, use_rhyme = False, use_syllables = False, union_or_intersection = 'intersection')
    # _, to_replace, words = PR.process(text, 10)
    # print('pos:', ', '.join(to_replace), ''.join(words))

    # PR = poem_replacer('lexicon.json', use_pos = False, use_anagrams = True, use_rhyme = False, use_syllables = False, union_or_intersection = 'intersection')
    # _, to_replace, words = PR.process(text, 10)
    # print('anagrams:', ', '.join(to_replace), ''.join(words))

    # PR = poem_replacer('lexicon.json', use_pos = False, use_anagrams = False, use_rhyme = True, use_syllables = False, union_or_intersection = 'intersection')
    # _, to_replace, words = PR.process(text, 10)
    # print('rhyme:', ', '.join(to_replace), ''.join(words))

    # PR = poem_replacer('lexicon.json', use_pos = False, use_anagrams = False, use_rhyme = False, use_syllables = True, union_or_intersection = 'intersection')
    # _, to_replace, words = PR.process(text, 10)
    # print('syllables:', ', '.join(to_replace), ''.join(words))

    # PR = poem_replacer('lexicon.json', use_pos = True, use_anagrams = True, use_rhyme = False, use_syllables = False, union_or_intersection = 'intersection')
    # _, to_replace, words = PR.process(text, 10)
    # print('pos, anagrams, intersection:', ', '.join(to_replace), ''.join(words))

    # PR = poem_replacer('lexicon.json', use_pos = True, use_anagrams = True, use_rhyme = False, use_syllables = False, union_or_intersection = 'union')
    # _, to_replace, words = PR.process(text, 10)
    # print('pos, anagrams, union:', ', '.join(to_replace), ''.join(words))

    PR = poem_replacer('lexicon.json', min_wn_distance=1, max_wn_distance=5)
    _, to_replace, words = PR.process(text, 10)
    print('wn_distances:', ', '.join(to_replace), ''.join(words))


if __name__ == '__main__':
    main()