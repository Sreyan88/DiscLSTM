import pandas as pd
import json 
import sys

split = "test"
dataset = "DailyDialog"

file = json.load(open('data/{}/{}_data_roberta_v2.json.feature'.format(dataset,split),'rb'))

list_of_dicts = []
for i,dialogue in enumerate(file):
    sample = {}
    edus = []
    num_texts_in_conversation = 0
    for conversation in dialogue:
        num_texts_in_conversation+=1
        utterance = conversation['text']
        speaker = conversation['speaker']
        edus.append({'speaker':speaker,'text':utterance})
    sample['edus'] = edus
    sample['id'] = str(i)
    list_of_dicts.append(sample)
    relations = []
    for i in range(num_texts_in_conversation):
        for j in range(i+1,num_texts_in_conversation):
            rel = {}
            rel['type'] = "Elaboration"
            rel['x'] = i
            rel['y'] = j
            relations.append(rel)
    sample['relations'] = relations


with open('{}_{}.json'.format(dataset,split),'w') as f:
    data = json.dumps(list_of_dicts)
    f.write(data)


