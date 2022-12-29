import spacy
import datamuse
import os
import time
from random import sample

BEGIN_TOKEN = '#BEGIN'
END_TOKEN = '#END'
REPLACE_TOKEN = '#REPLACE'
dm = datamuse.Datamuse()

def get_tokens(text):
    nlp = spacy.load("en_core_web_md")
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
    
def main(filename):
    start = time.time()
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, f'poems/{filename}')
    f = open(filepath, 'r')
    lines = f.readlines()
    data = '\n'.join(lines)
    f.close()

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
            print(f'Replacing {token.text}...')
            options = dm.words(md='p',ml=token.text,max=20)
            options = [o for o in options if 'tags' in o and spacyPOS_to_datamusePOS(token.pos_) in o['tags']]
            print(f'Num candidates: {len(options)}')
            if len(options) > 0:
                chosen = sample(options, 1)
                newWord = f'{chosen[0]["word"]}[#ORIGINAL_{token.text}]'
            else:
                newWord = f'{token.text}[#ORIGINAL_{token.text}]'
        
        out_words.append(f'{newWord}{token.whitespace_}')

    outFilename = filename.replace('.txt',f'_{time.time()}.txt')
    outFilepath = os.path.join(dirname, f'poems/generated/{outFilename}')
    fout = open(outFilepath, 'x')
    fout.write(''.join(out_words).replace('\n\n','\n'))
    fout.close()
    
    end = time.time()
    print(f'Total time elapsed: {end-start} seconds ({(end-start)/60} minutes)')

if __name__ == '__main__':
    main('stein-tender-buttons.txt')
    main('shakespeare-translations.txt')