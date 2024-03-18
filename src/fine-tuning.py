'''
References:
- https://docs.labelbox.com/reference/export-text-annotations
- https://www.linkedin.com/pulse/fine-tuning-open-ai-gpt-3-transformer-model-custom-dataset-hamid/
- https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
- https://platform.openai.com/docs/models/overview
'''
import os
import labelbox as lb
import re
import json
from openai import OpenAI

ai = OpenAI(organization=os.getenv("OPENAI_ORGANIZATION"), api_key=os.getenv("OPENAI_API_KEY"))

POEM_PROJECT_ID = os.getenv("LABELBOX_POEM_PROJECT_ID")

# Get labels from Labelbox
def getLabels():
    LB_API_KEY = os.getenv("LABELBOX_API_KEY")
    lb_client = lb.Client(api_key=LB_API_KEY)
    # print(next(lb_client.get_projects()))

    project = lb_client.get_project(POEM_PROJECT_ID)
    # labels = project.label_generator()
    labels = project.labels()
    # print(project.labels())
    label = next(labels)
    # url = label.data.url
    # annotations = label.annotations
    # print(annotations[0])
    print(label)

# [LEGACY] Generate new line of poetry using GPT-3.5 with prompt engineering
def callGPT35ChatCompletionModel(inputext):
    # Prompt engineering
    BASE_INSTRUCTION = 'Analyze the writing style of the following segment of poetry inside the square brackets and produce an edited version where each noun, verb, adjective, and adverb is replaced with a related word, synonym, homophone, or antonym. Respond with only the edited version, without annotation or explanation.'
    # BASE_INSTRUCTION = 'Analyze the writing style of the following segment of poetry inside the square brackets and produce an edited version where each noun, verb, adjective, and adverb is replaced with a related word, synonym, homophone, or antonym.'
    # BASE_INSTRUCTION = 'Analyze the writing style of the following segment of poetry inside the square brackets and produce a new version of he poem in the same style.'

    # Built in "GPT-3.5" model
    content = f'\n\n##\n\n{BASE_INSTRUCTION} [{inputext}]'
    print(content)
    response = ai.chat.completions.create(model='gpt-3.5-turbo',
                                messages=[{'role': 'user', 'content': content}])
    print(response)
    return response

# [LEGACY] Generate poem variation using regular GPT-3.5 with prompt engineering
def generateGPT35Variation(poem, segemnt=True):
    poemText = ' '.join([line.strip() for line in poem.split('\n')])
    # print(poemText)
    
    if(segemnt==True):
        sentences = poemText.split('.')
        variation = []
        line = ''
        for i in range(0,len(sentences)):
            s = sentences[i]
            print(f'Sentence {i}/{len(sentences)}: {len(s)}')
            line += s
            if len(line.strip()) > 800 or i == len(sentences)-1:
                res = callGPT35ChatCompletionModel(line)
                variation.append(res["choices"][0]["message"]["content"])
                line = ''
                # print(f'output: {res["choices"][0]["message"]["content"]}')

        return '\n'.join(variation)
    else:
        res = callGPT35ChatCompletionModel(poemText)
        return res["choices"][0]["message"]["content"]

# Generate training data for first round of curie fine tuning as text generation problem.
# Prompt: original line. Completion: New line
def fineTune1():
    # Get Labels from Labelbox ndjson exported file
    dirname = os.path.dirname(__file__)
    filePath = os.path.join(dirname, 'poems/labels/modernist-labels-good.ndjson')
    with open(filePath) as file:
        lines = [line.rstrip() for line in file]

    # Get good labels
    good_poems = []
    for poem in lines:
        obj = json.loads(poem)
        labels = obj['projects'][POEM_PROJECT_ID]['labels'][0]['annotations']['objects']
        good_labels = list(filter(lambda x: x['name']=='Good Line',labels))
        poemFilename = obj['data_row']['external_id']
        poemFilePath = os.path.join(dirname, f'poems/generated/direct-replace/{poemFilename}')
        try:
            with open(poemFilePath, 'r') as file:
                poem = file.read()
                # poemText = ' '.join([line.strip() for line in poem.split('\n')])
        except:
            continue

        # Format labels into jsonl prompts
        for label in good_labels:
            labeledLine = poem[label['location']['start'] : label['location']['end']+1].replace('\n','\\n')
            prompt = re.sub(r'\w+\[#ORIGINAL_([^\]]+)\]', r'\1', labeledLine)
            completion = re.sub(r'\[#ORIGINAL_[^\]]+]', '', labeledLine)
            # print(labeledLine)
            # print(prompt)
            # print(completion)
            trainingLine = '{' +  f'"prompt": "{prompt}\\n\\n###\\n\\n", "completion": " {completion}\\n"' + '}'
            # print(trainingLine)
            good_poems.append(trainingLine)
    print(len(good_poems))
    # print('\n'.join(good_poems))

    outFilePath = os.path.join(dirname, 'poems/labels/modernist-labels-good-training.jsonl')
    # fout = open(outFilePath, 'x')
    # fout.write('\n'.join(good_poems))
    # fout.close()

# [DEPRECATED] Generate poem variation using fine-tuned curie model for text generation.
# This model is no loner supported by OpenAI.
def generateFineTune1Variation(poem):
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
    print(''.join(newLines))

# Generate training data for second round of curie fine tuning as classification problem.
# Prompt: original line & new line. Completion: Good/Mediocre/Bad line
def fineTune2():
    # Get Labels from Labelbox ndjson exported file
    dirname = os.path.dirname(__file__)
    filePath = os.path.join(dirname, 'poems/labels/modernist-labels.ndjson')
    with open(filePath) as file:
        lines = [line.rstrip() for line in file]

    # Get good labels
    training_data = []
    for poem in lines:
        obj = json.loads(poem)
        labels = obj['projects'][POEM_PROJECT_ID]['labels'][0]['annotations']['objects']
        poemFilename = obj['data_row']['external_id']
        poemFilePath = os.path.join(dirname, f'poems/generated/direct-replace/{poemFilename}')
        try:
            with open(poemFilePath, 'r') as file:
                poem = file.read()
        except:
            continue

        # Format labels into jsonl prompts
        for label in labels:
            labeledLine = poem[label['location']['start'] : label['location']['end']+1].replace('\n','\\n')
            original = re.sub(r'\w+\[#ORIGINAL_([^\]]+)\]', r'\1', labeledLine).replace('"',"'")
            generated = re.sub(r'\[#ORIGINAL_[^\]]+]', '', labeledLine).replace('"',"'")
            original = re.sub(r'\\(?!n)','\\\\n',original)
            generated = re.sub(r'\\(?!n)','\\\\n',generated)
            # print(labeledLine)
            # print(original)
            # print(generated)
            trainingLine = '{' +  f'"prompt": "<original>{original}</original> : <generated>{generated}</generated>\\n\\n###\\n\\n", "completion": " {label["name"]}\\n"' + '}'
            # print(trainingLine)
            training_data.append(trainingLine)
    print(len(training_data))
    # print('\n'.join(training_data))

    outFilePath = os.path.join(dirname, 'poems/labels/ft2-modernist-labels-training_5000.jsonl')
    fout = open(outFilePath, 'x')
    fout.write('\n'.join(training_data[:5000]))
    fout.close()

# Create fine-tune job using training data files (already uploaded) generated using fineTune2()
# to fine-tune davinci-002 model (GPT-3)
def createFineTune():
    ai.fine_tuning.jobs.create(
        validation_file="file-dgYg5o2oys0H3vbdRnjx0CuT",
        training_file="file-AQpJv1IHrCjKCXWCWHrc6Jmm",
        model="davinci-002"
    )

if __name__ == '__main__':
    ftJobId = 'ftjob-BvPMxKfJxcnP2YsDQKqJalGC'
    print(ai.fine_tuning.jobs.retrieve(ftJobId))
    print(ai.fine_tuning.jobs.list_events(fine_tuning_job_id=ftJobId, limit=3))

    