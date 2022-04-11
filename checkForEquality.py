import json

dataset_name = "MELD"
split = "test"

with open('data/%s/%s_data_roberta_v9.json.feature'%(dataset_name, split), encoding='utf-8') as f:
    raw_data = json.load(f)
with open('data/dependency_graphs/meld_preds_%s.json'%(split), encoding='utf-8') as f:
    graph_uv = json.load(f)


def check_data_equality(graph_uv,raw_data):
    for idx in range(len(graph_uv)):
        if len(graph_uv[idx]["edus"])!=len(raw_data[idx]):
            print(idx)
            for d1,d2 in zip(graph_uv[idx]["edus"],raw_data[idx]):
                t1 = d1["text"]
                t2 = d2["text"]
                print(t1)
                print(t2)

check_data_equality(graph_uv,raw_data)