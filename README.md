# lgg-biomarker-prediction
 All the code and data visualizations

In this repository, you can see code examples of the following:

1) UMAP projections, which are useful for transforming multidimensional data into two dimensional plots to show groups of data.
2) XGBoost and Random Forests, which if you plan to implement, it is nice to have example code.
3) TCGA data and preprocessing, which is useful if you want to use TCGA data in the future–by going through this repository, you can save a lot of time getting useable data like mutations.

## When are UMAP Projections Useful?

In general, most of the stuff you would see here is useful for big data kind of projects. The UMAP projection is similar in function to principal component analysis (PCA), and t-SNE. I was actually going to use t-SNE to represent my data, but t-SNE actually could not capture the essence of my data. So when determining when to use t-SNE vs UMAP projections, you should really consider how weird and non-uniform your data is. Try both, see what is better. This process involves a lot of patience and you kinda just have to play with the hyperparameters till you get something you like.

## THE GOAT

towardsdatascience (blog)

## When to use Random Forests vs XGBoost?

So what you will see a lot everywhere is that XGBoost is used a lot because of its really good performance (a lot of people use XGBoost to win Kaggle competitions!). But the main problem with XGBoost is that it takes a long time. For my case, I just wanted feature importances, which is a pretty relative metric so using random forests was great for this and it really allowed me to speed up the process.

## Use Random Forests' Feature Importances!

When you use any kind of model, see if there is any sort of metric about feature importances–it will make a really cool graph/figure for your project!
