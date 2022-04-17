import json 

datasets = ['IEMOCAP','MELD','DailyDialog','EmoryNLP']
sets = ['train','dev','test']

for dataset in datasets:
    print(f'{dataset}')
    if dataset in ['IEMOCAP','DailyDialog']:
        for s in sets:
            file = open(f'data/{dataset}/{s}_data_roberta_v2.json.feature','rb')
            f = json.load(file)
            dialogs = len(f)
            utterances = sum([len(x) for x in f])
            print(f' {s} : Conversations {dialogs} Utterances {utterances}')
    else:
        for s in sets:
            file = open(f'data/{dataset}/{s}_data_roberta_v9.json.feature','rb')
            f = json.load(file)
            dialogs = len(f)
            utterances = sum([len(x) for x in f])
            print(f' {s} : Conversations {dialogs} Utterances {utterances}')
