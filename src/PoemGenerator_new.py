import spacy
import datamuse
import os
import time
import datetime
import concurrent.futures
from random import sample
from string import punctuation

BEGIN_AUTHOR_TOKEN = '#BEGIN_AUTHOR#'
END_AUTHOR_TOKEN = '#END_AUTHOR#'
BEGIN_TOKEN = '#BEGIN\n'
END_TOKEN = '#END\n'
REPLACE_TOKEN = '#REPLACE'
dm = datamuse.Datamuse()
nlp = spacy.load("en_core_web_md")

def get_tokens(text):
    doc = nlp(text)
    tokens = [token for token in doc]
    content_tokens = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
    return tokens, content_tokens               

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

def read_poem(filename):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, f'poems/{filename}')
    f = open(filepath, 'r')
    lines = f.readlines()
    data = '\n'.join(lines)
    f.close()
    return data

def save_poem(wordList, filename):
    dirname = os.path.dirname(__file__)
    outFilepath = os.path.join(dirname, f'poems/generated/{filename}')
    fout = open(outFilepath, 'x')
    fout.write(''.join(wordList).replace('\n\n','\n'))
    fout.close()

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

def run_experiments_on_poem():
    # replace_marked_words('stein-tender-buttons.txt')
    # replace_marked_words('shakespeare-translations.txt')

    filename = 'modernist-poems.txt'
    # percent_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # replacement_options = [['ml'], ['sp'], ['rel_trg'], ['rel_cns'], ['rel_hom'], ['ml','sp'], ['ana'], ['ml','ana']]
    
    percent_options = [100]
    replacement_options = [['ml'],['rel_trg']]
    
    args = [(filename, percent, repl) for percent in percent_options for repl in replacement_options]

    # Run all percent and replacement type experiments in parallel
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Using lambda, unpacks the tuple (*a) into hello(*args)
        executor.map(lambda a: replace_random_words(*a), args)
    end = time.perf_counter()

    print(f'Total time elapsed: {end-start} seconds ({(end-start)/60} minutes)')

# New 4/12
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

def run_author_poem(authorName, poem, idx):
    print(f'{authorName} {idx}')
    start = time.time()
    tokens, content_tokens = get_tokens(poem)
    replacement_options = [['ml'],['rel_trg']]
    args = [(authorName, idx, tokens, content_tokens, repl) for repl in replacement_options]

    with concurrent.futures.ThreadPoolExecutor() as executor3:
        executor3.map(lambda a: replace_all_random_words(*a), args)
        # for res in pool:
        #     print(res)

    end = time.time()

    print(f'\tDone {authorName} poem {idx} variations: {end-start} seconds ({(end-start)/60} minutes)')

def run_author_poems(authorText):
    start = time.time()
    poems = authorText.split(BEGIN_TOKEN)
    authorName = poems[0].strip()
    args = [(authorName, f'{BEGIN_TOKEN}\n{poems[i]}', i) for i in range(1,len(poems[1:]))]
    
    # print(f'Num {authorName} poems: {len(args)}')
    with concurrent.futures.ThreadPoolExecutor() as executor2:
        executor2.map(lambda a: run_author_poem(*a), args)
        # for res in pool:
        #     print(res)

    end = time.time()
    print(f'DONE {authorName}: {end-start} seconds ({(end-start)/60} minutes)')

def run_experiments_on_poems():
    start = time.time()
    print('Reading file...')
    filename = 'modernists_ascii.txt'
    data = read_poem(filename)
    authors = data.split(BEGIN_AUTHOR_TOKEN)
    
    print('Running poems...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = executor.map(run_author_poems, authors[1:])
        # for res in pool:
        #     print(res)
    end = time.time()
    print(f'DONE WHOLE FILE: {end-start} seconds ({(end-start)/60} minutes)')

if __name__ == '__main__':
    # run_experiments_on_poem()
    run_experiments_on_poems()