from nltk.tokenize import sent_tokenize
import os
import spacy
import labelbox as lb
import re
import time
import concurrent.futures
from openai import OpenAI
from random import sample
from version1 import generateNVariations, generateNVariations_parallel

ai = OpenAI(
    organization=os.getenv("OPENAI_ORGANIZATION"), api_key=os.getenv("OPENAI_API_KEY")
)

# Get classification label for given poem line and its generated variation using fine tuned model


def getLineCategory(original, generated, version="v2"):
    # FINE_TUNED_MODEL_2 = 'curie:ft-personal-2023-09-01-01-47-25' # 1000 training
    # FINE_TUNED_MODEL_3 ='curie:ft-personal-2023-09-01-03-25-31' # 5000 training
    PROMPT = f"<original>{original}</original> : <generated>{generated}</generated>\n\n###\n\n"
    if version == "v2":
        res = ai.completions.create(
            model="ft:davinci-002:personal::93yZODMm", prompt=PROMPT
        )
        category = res.choices[0].text.split("Line\n", 1)[0]
    elif version == "v3":
        category = version3(PROMPT)
    else:
        category = None
    # print(f'{category}: {generated}')
    return category


def version3(prompt):
    system_prompt = "You are a poetry analyzer. You will be given a pair of poem segments. One segment will be labelled as <original> and the other will be labeled as <generated>. Your job is to analyze the writing style of the given segments of poetry and determine whether the <generated> poem segment is a Good, Mediocre, or Bad alternative to the <original> segment. Respond with only the label."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    completion = ai.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal::B5ohvIap", messages=messages
    )

    return completion.choices[0].message.content.split("Line")[0].strip()


# Generate poem variation using only good/mediocre lines, classified using fine-tuned model
# Generate using 6 variations: 4 means_like, 1 triggered_by, 1 meaans_like & triggered_by


def generateFineTune2Variation(poem, version="v2"):
    start = time.time()
    originalLines = [line.strip() for line in poem.split("\n")]
    mls = generateNVariations(poem, 4, ["ml"])
    rts = generateNVariations(poem, 1, ["rel_trg"])
    ml_rts = generateNVariations(poem, 1, ["ml", "rel_trg"])
    lines1 = [line.strip() for line in mls[0].split("\n")]
    lines2 = [line.strip() for line in mls[1].split("\n")]
    lines3 = [line.strip() for line in mls[2].split("\n")]
    lines4 = [line.strip() for line in mls[3].split("\n")]
    lines5 = [line.strip() for line in rts[0].split("\n")]
    lines6 = [line.strip() for line in ml_rts[0].split("\n")]

    newLines = []
    for l in range(len(originalLines)):
        print(f"Line {l}")
        line = originalLines[l]
        line1 = (
            re.sub(r"\[#ORIGINAL_[^\]]+]", "", lines1[l]).replace('"', "'")
            if l < len(lines1)
            else line
        )
        line2 = (
            re.sub(r"\[#ORIGINAL_[^\]]+]", "", lines2[l]).replace('"', "'")
            if l < len(lines2)
            else line
        )
        line3 = (
            re.sub(r"\[#ORIGINAL_[^\]]+]", "", lines3[l]).replace('"', "'")
            if l < len(lines3)
            else line
        )
        line4 = (
            re.sub(r"\[#ORIGINAL_[^\]]+]", "", lines4[l]).replace('"', "'")
            if l < len(lines4)
            else line
        )
        line5 = (
            re.sub(r"\[#ORIGINAL_[^\]]+]", "", lines5[l]).replace('"', "'")
            if l < len(lines5)
            else line
        )
        line6 = (
            re.sub(r"\[#ORIGINAL_[^\]]+]", "", lines6[l]).replace('"', "'")
            if l < len(lines6)
            else line
        )

        label1 = getLineCategory(line, line1, version)
        label2 = getLineCategory(line, line2, version)
        label3 = getLineCategory(line, line3, version)
        label4 = getLineCategory(line, line4, version)
        label5 = getLineCategory(line, line5, version)
        label6 = getLineCategory(line, line6, version)

        labels = [
            {"line": line1, "label": label1},
            {"line": line2, "label": label2},
            {"line": line3, "label": label3},
            {"line": line4, "label": label4},
            {"line": line5, "label": label5},
            {"line": line6, "label": label6},
        ]
        good_labels = list(filter(lambda x: "good" in x["label"].lower(), labels))
        mediocre_labels = list(
            filter(lambda x: "mediocre" in x["label"].lower(), labels)
        )
        if len(good_labels) > 0:
            newLine = sample(good_labels, 1)[0]["line"]
            newLines.append(newLine + "\n")
        elif len(mediocre_labels) > 0:
            newLine = sample(mediocre_labels, 1)[0]["line"]
            newLines.append("[MEDIOCRE] " + newLine + "\n")
        else:
            newLines.append("-\n")

    print("".join(newLines))
    end = time.time()
    print(f"Total time: {end - start} seconds ({(end - start) / 60} minutes)")


# Parallelized version 2 algorithm
# Generate poem variation using only good/mediocre lines, classified using fine-tuned model
# Number of variations generated per replacement type are parameters


def generateFineTune2Variation_parallel(
    poem, nSyn, nRel, nAna, nSp, nCns, nHom, version="v2"
):
    totalNVars = nSyn + nRel + nAna + nSp + nCns + nHom
    print(f"Generating {totalNVars} variations...")
    start = time.time()
    nlp = spacy.load("en_core_web_md")
    originalLines = [line.strip() for line in poem.split("\n")]

    args = [
        (nlp, poem, nSyn, ["ml"]),
        (nlp, poem, nRel, ["rel_trg"]),
        (nlp, poem, nAna, ["ana"]),
        (nlp, poem, nSp, ["sp"]),
        (nlp, poem, nCns, ["rel_cns"]),
        (nlp, poem, nHom, ["rel_hom"]),
    ]
    poemVariations = []
    # Generate variations for each of the replacement types in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = executor.map(lambda a: generateNVariations_parallel(*a), args)
        for res in pool:
            for p in res:
                lines = [line.strip() for line in p.split("\n")]
                poemVariations.append(lines)

    print(f"Combining {totalNVars} variations...")
    newLines = [""] * len(originalLines)

    # Generate the final best output line from the variation options for the given original line
    def _processLine(idx, originalLine):
        labels = [{}] * len(poemVariations)

        # Get the category label (GOOD, MEDIOCRE, BAD) for a given line variation compared to the original
        def _processLineVariation(variationIdx, originalLine, variation):
            variationLine = (
                variation[idx].replace('"', "'")
                if idx < len(variation)
                else originalLine
            )
            cleanLine = re.sub(r"\[#ORIGINAL_[^\]]+]", "", variationLine)
            label = getLineCategory(originalLine, cleanLine, version)
            labels[variationIdx] = {"line": variationLine, "label": label}

        # Get the label for each poem variation ion parallel
        args = [
            (i, originalLine, variation) for i, variation in enumerate(poemVariations)
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pool = executor.map(lambda a: _processLineVariation(*a), args)

        # Find the good and mediocre labels
        good_labels = list(filter(lambda x: "good" in x["label"].lower(), labels))
        mediocre_labels = list(
            filter(lambda x: "mediocre" in x["label"].lower(), labels)
        )
        # Determine the final output line. Try random good label, then mediocre, or - placeholder
        if len(good_labels) > 0:
            newLine = sample(good_labels, 1)[0]["line"]
            newLines[idx] = newLine + "\n"
        elif len(mediocre_labels) > 0:
            newLine = sample(mediocre_labels, 1)[0]["line"]
            newLines[idx] = newLine + "\n"

    # Generate each output line in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pool = executor.map(lambda a: _processLine(*a), list(enumerate(originalLines)))

    # Combine all output lnes into a final output variation
    print("\n" + "".join(newLines))
    end = time.time()
    print(f"Total time: {end - start} seconds ({(end - start) / 60} minutes)")


def sentenceTokenization_nltk(poem):
    # NLTK
    return sent_tokenize(poem)


def sentenceTokenization_spacy(poem):
    # spaCy
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("sentencizer")
    doc = nlp(poem)
    sentence_list = []
    for sentence in doc.sents:
        sentence_list.append(sentence.text)
    return sentence_list


def sentenceTokenization_spacy2(poem):
    # spaCy (clauses)
    nlp = spacy.load("en_core_web_md")
    doc = nlp(poem)
    seen = set()  # keep track of covered words
    chunks = []
    for sent in doc.sents:
        heads = [cc for cc in sent.root.children if cc.dep_ == "conj"]

        for head in heads:
            words = [ww for ww in head.subtree]
            for word in words:
                seen.add(word)
            chunk = " ".join([ww.text for ww in words])
            chunks.append((head.i, chunk))

        unseen = [ww for ww in sent if ww not in seen]
        chunk = " ".join([ww.text for ww in unseen])
        chunks.append((sent.root.i, chunk))

    chunks = sorted(chunks, key=lambda x: x[0])
    sentence_list = [c for i, c in chunks]

    return sentence_list


if __name__ == "__main__":
    poem = """
        As the dead prey upon us,
        they are the dead in ourselves,
        awake, my sleeping ones, I cry out to you,
        disentangle the nets of being!

        I pushed my car, it had been sitting so long unused.
        I thought the tires looked as though they only needed air.
        But suddenly the huge underbody was above me, and the rear tires
        were masses of rubber and thread variously clinging together
        """
    nSyn = 2  # Similar meaning
    nRel = 0  # Related words
    nAna = 0  # Anagrams
    nSp = 0  # Similarly spelled
    nCns = 0  # Consonant match
    nHom = 0  # Homophones
    generateFineTune2Variation_parallel(poem, nSyn, nRel, nAna, nSp, nCns, nHom, "v3")

    # cleanPoem = " ".join([l.strip() for l in poem.split("\n") if l.strip() != ""])
    # # sents = sentenceTokenization_nltk(cleanPoem)
    # sents = sentenceTokenization_spacy(cleanPoem)
    # # sents = sentenceTokenization_spacy2(cleanPoem)
    # for s in sents:
    #     print(s)
