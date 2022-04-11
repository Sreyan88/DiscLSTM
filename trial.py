# import json 
# import glob
# import pandas as pd

# sets = ['DailyDialog','EmoryNLP','IEMOCAP','MELD']

# for s in sets:
#     connections = []
#     for filename in glob.glob(f'data/dependency_graphs/{s}_preds*'):
#         f = open(filename,'r')
#         jsonified = json.load(f)
#         for conv in jsonified:
#             connections.append(len(conv['relations']))
#     ser = pd.Series(connections)
#     print(s)
#     print(ser.describe())
import pandas as pd
file = pd.read_csv('lab.csv')
print(file.shape)
file = file[file['Label']!='xxx']
print(file.shape)
