## DiscLSTM

PyTorch Code for [A Discourse Aware Sequence Learning Approach for Emotion Recognition in Conversations](https://arxiv.org/abs/2203.16799)

### DiscLSTM Cell Representation:

![DiscLSTM Cell Designed :](./images/Model_image.jpg "This LSTM Cell Uses both the graph embeddings and text embeddings").

### Preparation:
Before running the model, download the following files for 4 datasets: 
- Extracted Utterance Features : https://drive.google.com/file/d/1R5K_2PlZ3p3RFQ1Ycgmo3TgxvYBzptQG/view?usp=sharing 
- Dependency Graphs : https://drive.google.com/drive/folders/14fMl_APuJ9S0y-2vwewzlC64_nYj7mum?usp=sharing

### Running the models:

You can train the models with the following codes:

```
python run.py --dataset DATASET --gnn_layers 2 --lr 0.0005 --batch_size 16 --epochs 30 --dropout 0.2
```

- 'DATASET' can be chosen from : [IEMOCAP,MELD,EmoryNLP,DailyDialog]
- 'gnn_layers' : choose the number of layers you want for Graph Embedded Representation

## Cite

If you find this work useful, please do cite our paper:  
```
@misc{https://doi.org/10.48550/arxiv.2203.16799,
  doi = {10.48550/ARXIV.2203.16799},
  url = {https://arxiv.org/abs/2203.16799},
  author = {Ghosh, Sreyan and Srivastava, Harshvardhan and Umesh, S.},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Discourse Aware Sequence Learning Approach for Emotion Recognition in Conversations},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```