# PDSum: Prototype-driven Continuous Summarization of Evolving Multi-document Sets Stream
Presented at WWW'23 [[Paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583371)]

## Task: EMDS (Evolving Multi-document Sets Stream Summarization)
![EMDS](https://github.com/cliveyn/PDSum/blob/main/EMDS_fig.jpg)

## How to run

### Packages
See requirements.txt (pip install -r requirements.txt)

### Datasets
Go to [link](https://www.dropbox.com/sh/0mc7i10qype7og6/AAAARaHV_UFFc6J70YXTwGDIa?dl=0) for the preprocessed data sets and put them under a "datasets" folder
(The original datasets can be found from [WCEP](https://github.com/complementizer/wcep-mds-dataset) and [W2E](https://github.com/smutahoang/w2e))

#### Sample (WCEP)
![dataset](https://github.com/cliveyn/PDSum/blob/main/dataset_sample.jpg)
For your own dataset, please preprocess the dataset in JSON file with the following columns
- date: the publish date of a document
- sentences: [s1, s2, .., sn] - a list of sentences in a document
- sentence_tokens: [[t1, t2, ..tn],[t1,...]] - list of tokens in each sentence
- sentence_counts: the number of sentences in a document
- Query: the identifier for a document set (e.g., type/category/query)


### Usage
Refer to PDSum_run_example.ipynb and set the datasets and hyperparameters (default values are given), and run the notebook as instructed.

## Cite
```
@inproceedings{pdsum,
  title={PDSum: Prototype-driven Continuous Summarization of Evolving Multi-document Sets Stream},
  author={Yoon, Susik and Chan, Hou Pong and Han, Jiawei},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={1650--1661},
  year={2023}
}
```
