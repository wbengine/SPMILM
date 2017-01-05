# TRF top level design  {#top-level}
### 1.Features:
See module [Feature](group__feature.html)
- Class Feat：store all the features that has occurred. Each feature is indexed with an integer. The class Feat has an member array of class: FeatTable.
       + Class FeatTable：Store features of a set of similar feature templates. For instance, a FeatTable with constructing parameter w[1:3] will store the features of the following templates:（w_0, w_-1w_0, w_-2w_-1w_0）. A FeatTable object use a member object Trie<VocabID, int> to store those similar fetures. The key of the Trie is IDs of the words, and the value of the Trie is the index for the corresponding feature. 
       + class FeatStyle：A specific feature temple. Represented by a struct Field. 

- Class Seq：A word sequence which represent a sentence. The class Seq has a member variable of type Mat, with index of each word as the first row, classes of each word as second row. 

### 2.The model and sample algorithms：
See module [Model](group__model.html)
- class Model：Define and update the parameters. Define the sample algorithms：
 + LocalJump(Seq &seq) Propose and accept the length of next sentence.
 + MarkovMove(Seq &seq) Gibbs sampling each word of a fixed-length sentence. \n

Class Model has member objects of following classes:
	- class Corpus：read the training/validation set and build a corpus
	- class AlgNode： implementation of forward-backward algorithm

### 3.Training of the model:
See module [SA Training](group__train.html)
- class SAfunc：Implementation of the SA training algorithm. \n
  For each iteration, commit a TRF sampling and obtain a set of sentences which obey the joint distribution of the pair (l, xl) calculated by current model parameters. Use the difference of E[f] with respect to empirical distribution and joint distribution as the update vector.\n
  class SAfunc has member objects of following classes：
       + class Model

- class SAtrain：Carry out the SA training \n
  class SAtrain has member objects of following classes：
	+ class SAfunc
	+ class LearningRate：adjust learning rate in respect to iteration times

### 4.Word Clustering：
See module [Word Cluster](group__cluster.html)
- class WordCluster：Cluster the words into several classes. The objective to minimize is dRes = dSumClassGram - 2 * dSumClass + m_dWordLogSum. Greedy method is used in each iteration. 

### 5.Data Structure：
See mudule [Data Strcture Used](group__struct.html)

### 6.System Tools：
See module [System Tools](group__system.html)
