import pandas as pd
import json 

file = pd.read_csv('MELD.Raw/dev_sent_emo.csv')

dialogues = file['Dialogue_ID'].unique()

list_of_dicts = []
for dialogue in dialogues:
    sample = {}
    conversation = file[file['Dialogue_ID']==dialogue]
    edus = []
    num_texts_in_conversation = 0
    for i,row in conversation.iterrows():
        num_texts_in_conversation+=1
        utterance = row['Utterance']
        speaker = row['Speaker']
        edus.append({'speaker':speaker,'text':utterance})
    sample['edus'] = edus
    sample['id'] = str(dialogue)
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


with open('meld_dev.json','w') as f:
    data = json.dumps(list_of_dicts)
    f.write(data)


