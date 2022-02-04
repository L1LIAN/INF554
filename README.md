# INF554
ML Kaggle challenge : H-index prediction

------------READ ME--------------------------------------------------------------------------

3 python files :

	* Analysis of abstracts.ipynb -> Linking papers related files to authors, and computing the relationship into a single file. Then using it to estimate h-indexes.

	* Other_algo.ipynb -> Tokenisation of abstracts using BERT, then neural network fitting and testing on them.

	* Code.py -> an edited and commented version of baseline adding and testing many more features. Baseline is divided in named-cells, which explain what they individually do


------More details on baseline----------------------------------------------------------

Our implementation of gensim glove uses the glove.6B.300d.txt, and it is needed to use baseline.

Some cells are not useful for our top kaggle score (they are marked with three stars) (located at the very end)

Other cells should only be used once to create new files, that we can then re-use (marked with one star). You should avoid re-using them as they are quite time consuming (before three-star-cells)

Proceed linearly and cell by cell when executing (some prints are included to help assess the code's execution)

In the MAIN cell, you can modify wich feature is going to be used :
graph based features (degree...) - set as True
gensim features (full 300-sized vectors, no pca) - set as True
gensim lite (same but only pca-ed features from gensim, number set as nc) - set as False
nber of papers feature (one of our costume features, detailed in the document) : set as True
