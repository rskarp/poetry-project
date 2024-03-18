import spacy
import datamuse
import os
import time
import datetime
import concurrent.futures
from openai import OpenAI
from random import sample
from string import punctuation

ai = OpenAI(organization=os.getenv("OPENAI_ORGANIZATION"), api_key=os.getenv("OPENAI_API_KEY"))

BEGIN_AUTHOR_TOKEN = '#BEGIN_AUTHOR#'
END_AUTHOR_TOKEN = '#END_AUTHOR#'
BEGIN_TOKEN = '#BEGIN\n'
END_TOKEN = '#END\n'
REPLACE_TOKEN = '#REPLACE'
dm = datamuse.Datamuse()
nlp = spacy.load("en_core_web_md")

# Tokenize poem using spaCy
def get_tokens(text):
    doc = nlp(text)
    tokens = [token for token in doc]
    content_tokens = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
    return tokens, content_tokens               

# Map part of speech label from spaCy label to Datamuse label
def spacyPOS_to_datamusePOS(pos):
    if pos == 'NOUN':
        return 'n'
    elif pos == 'VERB':
        return 'v'
    elif pos == 'ADJ':
        return 'adj'
    elif pos == 'ADV':
        return 'adv'
    else:
        return ''

# Read poem text from file
def read_poem(filename):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, filename)
    f = open(filepath, 'r')
    lines = f.readlines()
    data = '\n'.join(lines)
    f.close()
    return data

# Save given list of poetry lines to given filename
def save_poem(wordList, filename):
    dirname = os.path.dirname(__file__)
    outFilepath = os.path.join(dirname, filename)
    fout = open(outFilepath, 'x')
    fout.write(''.join(wordList).replace('\n\n','\n'))
    fout.close()

# Get replacement candidates using Datamuse API for the given word using the given replacement types
def get_candidates(token, replacement_types, max_results = 50):
    # print(f'Replacing {token.text}...')
    text = token.text.strip(punctuation)
    ml = text if 'ml' in replacement_types else None
    sp = f'//{text}' if 'ana' in replacement_types else text if 'sp' in replacement_types else None
    rel_trg = text if 'rel_trg' in replacement_types else None
    rel_cns = text if 'rel_cns' in replacement_types else None
    rel_hom = text if 'rel_hom' in replacement_types else None
    
    # Get words using Datamuse API
    options = dm.words(md='p', ml=ml, sp=sp, rel_trg=rel_trg, rel_cns=rel_cns, rel_hom=rel_hom, max=max_results)
    # Filter words by matching part of speech
    return [o for o in options if 'tags' in o and spacyPOS_to_datamusePOS(token.pos_) in o['tags']]

# [LEGACY] Replace marked words in the poem at the given filename using given replacement types
# Save the output poem to a new file
def replace_marked_words(filename, replacement_types=['ml'], max_options=50):
    start = time.time()
    data = read_poem(filename)

    out_words = []
    tokens, content_tokens = get_tokens(data.replace(REPLACE_TOKEN,'#REPLACE '))
    readingPoemContent = False
    print(f'Proccessing {len(tokens)} tokens')
    for i, token in enumerate(tokens):
        if i >= 2 and tokens[i-1].text == 'BEGIN' and tokens[i-2].text == '#':
            readingPoemContent = True
        elif i >= 2 and tokens[i-1].text == 'END' and tokens[i-2].text == '#':
            readingPoemContent = False
        toReplace = i >= 2 and tokens[i-1].text == 'REPLACE' and tokens[i-2].text == '#'
        newWord = token.text
        if readingPoemContent and toReplace:
            out_words = out_words[:-2]
            options = get_candidates(token, replacement_types, max_options)
            if len(options) < 5:
                print(f'Num candidates: {len(options)}, word: {token.text}')
            if len(options) > 0:
                chosen = sample(options, 1)
                newWord = f'{chosen[0]["word"]}[#ORIGINAL_{token.text}]'
            else:
                newWord = f'{token.text}[#ORIGINAL_{token.text}]'
        
        out_words.append(f'{newWord}{token.whitespace_}')

    outFilename = filename.replace('.txt',f'_{time.time()}.txt')
    save_poem(out_words, outFilename)
    
    end = time.time()
    print(f'Total time elapsed: {end-start} seconds ({(end-start)/60} minutes)')

# [LEGACY] Replace the given percent of random content words in the poem at the given filename
# using given replacement types. Save the output poem to a new file.
def replace_random_words(filename, percent_to_replace, replacement_types=['ml'], max_options=50):
    start = time.time()
    data = read_poem(filename)

    out_words = []
    tokens, content_tokens = get_tokens(data)
    number_to_replace = int(len(content_tokens) * (percent_to_replace / 100))
    # choose number_to_replace tokens to replace
    to_replace = sample(content_tokens, number_to_replace)
    readingPoemContent = False
    numPoems = 0
    print(f'Proccessing {len(tokens)} tokens')
    for i, token in enumerate(tokens):
        if i >= 2 and tokens[i-1].text == 'BEGIN' and tokens[i-2].text == '#':
            readingPoemContent = True
            numPoems += 1
            print(f'Proccessing poem #{numPoems}')
        elif i >= 2 and tokens[i-1].text == 'END' and tokens[i-2].text == '#':
            readingPoemContent = False
        newWord = token.text
        if readingPoemContent and token in to_replace:
            options = get_candidates(token, replacement_types, max_options)
            # print(f'Num candidates: {len(options)}')
            if len(options) <= 1:
                print(f'Num candidates: {len(options)}, word: {token.text}, type: {",".join(replacement_types)}')
            if len(options) > 0:
                chosen = sample(options, 1)
                newWord = f'{chosen[0]["word"]}[#ORIGINAL_{token.text}]'
            else:
                newWord = f'{token.text}[#ORIGINAL_{token.text}]'
        
        out_words.append(f'{newWord}{token.whitespace_}')

    curDatetime = datetime.datetime.now().strftime( '%Y-%m-%d-%H.%M.%S.%f')
    outFilename = filename.replace('.txt',f'_{percent_to_replace}_{"_".join(replacement_types)}_{curDatetime}.txt')
    print(outFilename)
    save_poem(out_words, outFilename)
    
    end = time.time()
    print(f'Experiment time elapsed ({outFilename}): {end-start} seconds ({(end-start)/60} minutes)')

# [LEGACY] Generate poems from modernist-poems using various replacement percentages & replacement types
def run_experiments_on_poem():
    filename = 'modernist-poems.txt'

    percent_options = [10, 50, 100]
    replacement_options = [['ml'],['ml'],['ml'],
                           ['rel_trg'],['rel_trg'],['rel_trg'],['rel_trg'],['rel_trg'],
                           ['rel_trg','ml'],['rel_trg','ml'],['rel_trg','ml'],['rel_trg','ml'],['rel_trg','ml']]
    
    args = [(filename, percent, repl) for percent in percent_options for repl in replacement_options]

    # Run all percent and replacement type experiments in parallel
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Using lambda, unpacks the tuple (*a) into hello(*args)
        executor.map(lambda a: replace_random_words(*a), args)
    end = time.perf_counter()

    print(f'Total time elapsed: {end-start} seconds ({(end-start)/60} minutes)')

# [LEGACY] Helper function for run_author_poem(). Replaces 100% of content words for given poem.
# Saves generated poem to new file.
def replace_all_random_words(authorName, idx, tokens, content_tokens, replacement_types=['ml'], max_options=50):
    start = time.time()

    out_words = []
    readingPoemContent = False
    for i, token in enumerate(tokens):
        if i >= 2 and tokens[i-1].text == 'BEGIN' and tokens[i-2].text == '#':
            readingPoemContent = True
        elif i >= 2 and tokens[i-1].text == 'END' and tokens[i-2].text == '#':
            readingPoemContent = False
        newWord = token.text
        if readingPoemContent and token in content_tokens:
            options = get_candidates(token, replacement_types, max_options)
            if len(options) <= 1:
                print(f'Num candidates: {len(options)}, word: {token.text.strip(punctuation)}, type: {",".join(replacement_types)}')
            if len(options) > 0:
                chosen = sample(options, 1)
                newWord = f'{chosen[0]["word"]}[#ORIGINAL_{token.text}]'
            else:
                newWord = f'{token.text}[#ORIGINAL_{token.text}]'
        
        out_words.append(f'{newWord}{token.whitespace_}')

    curDatetime = datetime.datetime.now().strftime( '%Y-%m-%d-%H.%M.%S.%f')
    outFilename = f'{authorName}-{idx}_100_{"_".join(replacement_types)}_{curDatetime}.txt'
    print(outFilename)
    save_poem(out_words, outFilename)
    
    end = time.time()
    print( f'\t\tPoem variation time elapsed ({outFilename}): {end-start} seconds ({(end-start)/60} minutes)')

# [LEGACY] Helper function for run_author_poems(). Generates and saves multiple versions of the given poem
# using various replacement types in parallel.
def run_author_poem(authorName, poem, idx):
    print(f'{authorName} {idx}')
    start = time.time()
    tokens, content_tokens = get_tokens(poem)
    replacement_options = [['ml'],['ml'],['ml'],
                           ['rel_trg'],['rel_trg'],['rel_trg'],['rel_trg'],['rel_trg'],
                           ['rel_trg','ml'],['rel_trg','ml'],['rel_trg','ml'],['rel_trg','ml'],['rel_trg','ml']]
    
    args = [(authorName, idx, tokens, content_tokens, repl) for repl in replacement_options]

    with concurrent.futures.ThreadPoolExecutor() as executor3:
        executor3.map(lambda a: replace_all_random_words(*a), args)
        # for res in pool:
        #     print(res)

    end = time.time()

    print(f'\tDone {authorName} poem {idx} variations: {end-start} seconds ({(end-start)/60} minutes)')

# [DEPRECATED] Helper function for run_author_poems(). Uses fine-tuned GPT-3 model for text generation
# to generate and save a new poem variation of the given poem.
# OpenAI no longer supports this model.
def run_author_poem_ai(authorName, poem, idx):
    print(f'{authorName} {idx}')
    start = time.time()
    FINE_TUNED_MODEL = 'curie:ft-personal-2023-07-11-08-25-05'

    originalLines = [line.strip() for line in poem.split('\n')]
    newLines = []
    for line in originalLines:
        PROMPT=f'{line} \n\n###\n\n'
        res = ai.completions.create(
        model=FINE_TUNED_MODEL,
        prompt=PROMPT)
        newLine = res['choices'][0]['text'].split("\n END", 1)[0].replace('\n','')
        newLines.append(newLine+'\n')
    # print(''.join(newLines))

    curDatetime = datetime.datetime.now().strftime( '%Y-%m-%d-%H.%M.%S.%f')
    outFilename = f'{authorName}-{idx}_100_AI_{curDatetime}.txt'
    print(outFilename)
    save_poem(''.join(newLines), outFilename)

    end = time.time()

    print(f'\tDone {authorName} poem {idx} variations: {end-start} seconds ({(end-start)/60} minutes)')

# [LEGACY] Helper function for run_experiments_on_poems(). Creates and saves poem variations for all poems
# by a single author. authorText must be formatted separating poems using the BEGIN_TOKEN and END_TOKEN
def run_author_poems(authorText):
    start = time.time()
    poems = authorText.split(BEGIN_TOKEN)
    authorName = poems[0].strip()
    # print(f'{authorName}: {len(poems)-1} poems')
    args = [(authorName, f'{BEGIN_TOKEN}\n{poems[i]}', i) for i in range(1,len(poems))]

    with concurrent.futures.ThreadPoolExecutor() as executor2:
        executor2.map(lambda a: run_author_poem_ai(*a), args)
        # for res in pool:
        #     print(res)

    end = time.time()
    print(f'DONE {authorName}: {end-start} seconds ({(end-start)/60} minutes)')

# [LEGACY] Reads originals.txt with our selected poems, with specific markers denoting author change and
# poem change. Generates and saves poem variations for each poem by each author in parallel.
def run_experiments_on_poems():
    start = time.time()
    print('Reading file...')
    filename = 'originals.txt'
    data = read_poem(filename)
    authors = data.split(BEGIN_AUTHOR_TOKEN)
    print(f'{len(authors)} authors')
    print('Running poems...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = executor.map(run_author_poems, authors[1:])
        # for res in pool:
        #     print(res)
    end = time.time()
    print(f'DONE WHOLE FILE: {end-start} seconds ({(end-start)/60} minutes)')

# Generate and save a single poem variation of the given poem text by replacing 100% of content words with
# similar meaning words.
def generatePoem(text):
    start = time.time()
    dm = datamuse.Datamuse()
    nlp = spacy.load("en_core_web_md")

    # getTokens(text)
    doc = nlp(text)
    tokens = [token for token in doc]
    content_tokens = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

    # replaceRandomWords
    replacement_types=['ml']
    max_options=50
    percent_to_replace = 100
    number_to_replace = int(len(content_tokens) * (percent_to_replace / 100))
    out_words=[]
    # choose number_to_replace tokens to replace
    to_replace = sample(content_tokens, number_to_replace)
    # numPoems = 0
    print(f'Proccessing {len(tokens)} tokens')
    for i, token in enumerate(tokens):
        newWord = token.text
        if  token in to_replace:
            options = get_candidates(token, replacement_types, max_options)
            # print(f'Num candidates: {len(options)}')
            if len(options) <= 1:
                print(f'Num candidates: {len(options)}, word: {token.text}, type: {",".join(replacement_types)}')
            if len(options) > 0:
                # print(len(options))
                chosen = sample(options, 1)
                newWord = f'{chosen[0]["word"]}[#ORIGINAL_{token.text}]'
            else:
                newWord = f'{token.text}[#ORIGINAL_{token.text}]'
        
        out_words.append(f'{newWord}{token.whitespace_}')

    curDatetime = datetime.datetime.now().strftime( '%Y-%m-%d-%H.%M.%S.%f')
    outFilename = f'_{percent_to_replace}_{"_".join(replacement_types)}_{curDatetime}.txt'
    print(outFilename)

    end = time.time()
    print(f'Total time elapsed: {end-start} seconds ({(end-start)/60} minutes)')
    print(''.join(out_words).replace('\n\n','\n'))

# Generate n variations of the given poem text by replacing 100% of content words with words of the
# given replacement types. The poems are returned and not saved to new files.  Used in version2.py.
def generateNVariations(text, n, replacement_types=['ml']):
    start = time.time()
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    tokens = [token for token in doc]
    content_tokens = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

    max_options=50
    percent_to_replace = 100
    number_to_replace = int(len(content_tokens) * (percent_to_replace / 100))
    poems = [[] for _ in range(n)]
    # choose number_to_replace tokens to replace
    to_replace = sample(content_tokens, number_to_replace)
    # numPoems = 0
    print(f'Proccessing {len(tokens)} tokens')
    for i, token in enumerate(tokens):
        newWords = [token.text for _ in range(n)]
        if  token in to_replace:
            options = get_candidates(token, replacement_types, max_options)
            # if len(options) <= 1:
            #     print(f'Num candidates: {len(options)}, word: {token.text}, type: {",".join(replacement_types)}')
            if len(options) > 0:
                chosenWords = sample(options, min([n,len(options)]))
                numChosen = len(chosenWords)
                newWords = [f'{chosenWords[i%numChosen]["word"]}[#ORIGINAL_{token.text}]' for i in range(n)]
            else:
                newWords = [f'{token.text}[#ORIGINAL_{token.text}]' for i in range(n)]

        for p in range(n):
            poems[p].append(f'{newWords[p]}{token.whitespace_}')

    end = time.time()
    print(f'Generated {n} variations in: {end-start} seconds ({(end-start)/60} minutes)')

    for p in range(n):
        poems[p] = ''.join(poems[p]).replace('\n\n','\n')

    return poems

# Parallelized version of generateNVariations(). Used in version2.py
# Generate n variations of the given poem text by replacing 100% of content words with words of the
# given replacement types. The poems are returned and may optionally be saved to new files.
def generateNVariations_parallel(nlp, text, nVars, replacement_types=['ml'], save=False, filenameBase=''):
    start = time.time()
    doc = nlp(text)
    tokens = [token for token in doc]
    content_tokens = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

    max_options=50
    percent_to_replace = 100
    number_to_replace = int(len(content_tokens) * (percent_to_replace / 100))
    poems = [['']*len(tokens) for _ in range(nVars)]
    # choose number_to_replace tokens to replace
    to_replace = sample(content_tokens, number_to_replace)
    # print(f'Proccessing {len(tokens)} tokens')

    def _processToken(idx, token):
        newWords = [token.text for _ in range(nVars)]
        if  token in to_replace:
            options = get_candidates(token, replacement_types, max_options)
            # if len(options) <= 1:
            #     print(f'Num candidates: {len(options)}, word: {token.text}, type: {",".join(replacement_types)}')
            if len(options) > 0:
                chosenWords = sample(options, min([nVars,len(options)]))
                numChosen = len(chosenWords)
                newWords = [f'{chosenWords[i%numChosen]["word"]}[#ORIGINAL_{token.text}]' for i in range(nVars)]
            else:
                newWords = [f'{token.text}[#ORIGINAL_{token.text}]' for i in range(nVars)]

        for p in range(nVars):
            poems[p][idx] = f'{newWords[p]}{token.whitespace_}'

    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = executor.map(lambda a: _processToken(*a), list(enumerate(tokens)))

    for p in range(nVars):
        if save == True:
            curDatetime = datetime.datetime.now().strftime( '%Y-%m-%d-%H.%M.%S.%f')
            outFilename = f'{filenameBase}_100_{"_".join(replacement_types)}_{curDatetime}.txt'
            print(outFilename)
            save_poem(poems[p], outFilename)
        poems[p] = ''.join(poems[p]).replace('\n\n','\n')

    end = time.time()
    print(f'Generated {nVars} variations in: {end-start} seconds ({(end-start)/60} minutes)')

    return poems

# [LEGACY] Generates multiple variations of each poem in the given file, using the given replacement types.
# The file must be formatted using specific tokens denoting the beginning & end of each poem and of each
# section of poems written by an author. Saves each generated poem to new file.
def generateVariationsFromFile(filename='poems/original/modernists_ascii.txt', nVarsPerType=1, replacementTypes=['ml'],outputPath='poems/generated'):
    start = time.time()
    nlp = spacy.load("en_core_web_md")
    data = read_poem(filename)
    authors = data.split(BEGIN_AUTHOR_TOKEN)
    print(f'{len(authors)} authors')
    print('Running poems...')

    def generateVariationsForAuthor(authorText):
        start = time.time()
        poems = authorText.split(BEGIN_TOKEN)
        authorName = poems[0].strip()
        print(f'{authorName}: {len(poems)-1} poems')
        poems = ['\n'.join([line.strip() for line in poem.split('\n')]) for poem in poems]
        args = [(nlp, poems[i],nVarsPerType,replacementTypes,True,f'{outputPath}/{authorName}-{i}') for i in range(1,len(poems))]

        with concurrent.futures.ThreadPoolExecutor() as executor2:
            executor2.map(lambda a: generateNVariations_parallel(*a), args)

        end = time.time()
        print(f'DONE {authorName}: {end-start} seconds ({(end-start)/60} minutes)')
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = executor.map(generateVariationsForAuthor, authors[1:])
    end = time.time()
    print(f'DONE WHOLE FILE: {end-start} seconds ({(end-start)/60} minutes)')

# Parallelized version of generatePoem().
# Generates a single variation of hte given poem using the given replacement types.
# Processes each content word in parallel.
# Returns the generated poem (does not save to file).
def createPoemVariation(text, replacement_types=['ml']):
    tokens, content_tokens = get_tokens(text)
    max_options = 50
    percent_to_replace = 100
    number_to_replace = int(len(content_tokens) * (percent_to_replace / 100))
    poem = ['']*len(tokens)
    # choose number_to_replace tokens to replace
    to_replace = sample(content_tokens, number_to_replace)

    def _processToken(idx, token):
        newWord = token.text
        if token in to_replace:
            options = get_candidates(token, replacement_types, max_options)
            if len(options) > 0:
                chosenWords = sample(options, 1)
                newWord = f'{chosenWords[0]["word"]}[#ORIGINAL_{token.text}]'
            else:
                newWord = f'{token.text}[#ORIGINAL_{token.text}]'

        poem[idx] = f'{newWord}{token.whitespace_}'
        return poem[idx]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = executor.map(lambda a: _processToken(*a),
                            list(enumerate(tokens)))

    poem = ''.join(poem).replace('\n\n', '\n')

    return poem


if __name__ == '__main__':
    poem = '''
        As the dead prey upon us,
        they are the dead in ourselves,
        awake, my sleeping ones, I cry out to you,
        disentangle the nets of being!'''

    p = createPoemVariation(poem)
    print(p)