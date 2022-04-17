import json
import re

def load_data_predict(filename, map_relations):
    print "Loading data:", filename
    f_in = open(filename)
    inp = f_in.readline()
    data = json.loads(inp)
    num_sent = 0
    cnt_multi_parents = 0
    for dialog in data:
        last_speaker = None
        turn = 0
        for edu in dialog["edus"]:
            edu["text_raw"] = edu["text"] + " "
            text = edu["text"]
            
            while text.find("http") >= 0:
                i = text.find("http")
                j = i
                while (j < len(text) and text[j] != ' '): j += 1
                text = text[:i] + " [url] " + text[j + 1:]
            
            invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
            for ch in invalid_chars:
                text = re.sub(ch, "", text)
            tokens = []
            cur = 0
            for i in range(len(text)):
                if text[i] in "',?.!()\": ":
                    if (cur < i):
                        tokens.append(text[cur:i])
                    if text[i] != " ":
                        if len(tokens) == 0 or tokens[-1] != text[i]:
                            tokens.append(text[i])
                    cur = i + 1
            if cur < len(text):
                tokens.append(text[cur:])
            tokens = [token.lower() for token in tokens]
            for i, token in enumerate(tokens):
                if re.match("\d+", token): 
                    tokens[i] = "[num]"
            edu["tokens"] = tokens
            
            if edu["speaker"] != last_speaker:
                last_speaker = edu["speaker"]
                turn += 1
            edu["turn"] = turn      
    f_in.close()
    cnt_edus, cnt_relations, cnt_relations_backward = 0, 0, 0
    for dialog in data:
        cnt_edus += len(dialog["edus"])
    print "%d dialogs, %d edus" % \
        (len(data), cnt_edus)
        
    return data

map_relations = {}
split = "train"
raw_data = load_data_predict('meld_train.json', map_relations)
with open('data/dependency_graphs/meld_preds_%s.json'%(split), encoding='utf-8') as f:
    graph_uv = json.load(f)

for idx in range(len(graph_uv)):
    if len(graph_uv[idx]["edus"])!=len(raw_data[idx]["edus"]):
        print(idx)
        for d1,d2 in zip(graph_uv[idx]["edus"],raw_data[idx]["edus"]):
            t1 = d1["text"]
            t2 = d2["text"]
            print(t1)
            print(t2)