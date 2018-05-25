# City Names Classification

In this project, I trained a model to classify cities of France by region, processing letters of city names sequentially. 

The first objective of this project was to get familiar with PyTorch modules nn and autograd. We don't get a tremendous accuracy after training a simple and shallow LSTM, but whith time and computational ressources, there is room for a lot of improvements. Just to name a few:

* increase the depth and number of layers of the LSTM
* use character embeddings
* use CNN layers first to encode local information
* fine tune the hyperparameters

The three CSV files used in this project can be found in the Open platform for French public data: https://www.data.gouv.fr/en/datasets/regions-departements-villes-et-villages-de-france-et-doutre-mer/
