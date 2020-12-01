# Extreme Multi-label Classification

In this project, we experimented with various extreme multi-label classification algorithms on a large-scale data set.

## Task
Extreme classification is a multi-label classification problem that annotates a data point with the most relevant subset of labels from an extremely large label set. 
It has wide applications in diverse areas such as dynamic search advertising, text classification, and recommender systems. 
The main technical challenges include improving the prediction accuracy and reducing the training time, prediction time and model size. 

## Data
In this project, we performed extreme multi-label classification on [EURLex-4K dataset](http://manikvarma.org/downloads/XC/XMLRepository.html), 
a collection of documents about European Union Law with 3993 categories. 

## Methods

We first applied traditional multi-label algorithms as baseline. 
There are two traditional methods that we tried:
1. Problem Transformation (Binary Relevance/Classifier Chain plus Traditional ML algorithms like RF/KNN)
2. Algorithm Adaption (Adapted KNN, SVM etc.)

We further implemented embedding-based models [Principal Label Space Transformation (PLST)](https://www.csie.ntu.edu.tw/~htlin/paper/doc/plst.pdf) 
and [Sparse Local Embeddings for Extreme Multi-label Classification (SLEEC)](https://papers.nips.cc/paper/5969-sparse-local-embeddings-for-extreme-multi-label-classification), 
and we modified existed algorithms for improvements. 

Finally, we focused on one of the leading one-vs-all based extreme classifiers [Partitioned Label Trees (Parabel)](http://manikvarma.org/pubs/prabhu18b.pdf). 

## Evaluation

### LRAP
We used [label ranking average precision (LRAP)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html) as our evaluation metric to assess label ranking performance. 
### Training Time
We also record training times to evaluate model efficiency. 

## Conclusion
The result shows that the Parabel achieves the highest LRAP score as well as the best training time 
among all the algorithms we experimented with.

## Contributors

* Man Jin (mj1637@nyu.edu)
* Florence Denglin Jiang (dj1369@nyu.edu)
* Hong Gong (hg1153@nyu.edu)
* Jacqueline Yuwei Wang (yw1854@nyu.edu)
* Yi Xu (yx2090@nyu.edu)
