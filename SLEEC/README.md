# SLEEC Model

We novelly modified the [SLEEC algorithm](https://papers.nips.cc/paper/5969-sparse-local-embeddings-for-extreme-multi-label-classification) 
to improve its performance and efficiency by:

1. use KD- tree to search for nearest neighbors to fight against curse of dimensionality; 

2. solve matrix completion problem with Alternating Least Square 
instead of the Singular Value Projection to learn low-rank embedding, 
which trades performance for computation efficiency; 

3. use Elastic Net instead of L1 norm as regularization term 
for our multi-linear regressors to gain more robustness while controlling model complexity.

## Reference 

We adapted a [Python implementation of SLEEC](https://github.com/xiaohan2012/sleec_python).

Also please refer to the original if needed:
[matlab implementation](http://manikvarma.org/code/SLEEC/download.html)
