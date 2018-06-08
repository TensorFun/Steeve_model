# Steeve_model

This repo is the development of Steeve's NLP model. The complete product is in [Steeve_bot](https://github.com/TensorFun/Steeve_bot.git) repo.


## Detail

In order to get a better performance, we want to train a classifier to predict user's suitable field. 
Therefore, here we used SVM to train our model. First, we utilize a filter to retrieve programming languages, and then use these extracted programming languages as features, and the label is the field the post is from.

## Methods

#### Train Model
1. Retrieve programming languages(PLs) from the posts.
2. Convert these PLs into 300 dimension vectors.
3. Sum PLs from the same post as its feature, and label is the field which the post is from.
4. Put data into SVM.

#### Apply Model
1. Retrieve PLs from user's input.
2. Convert these PLs into 300 dimension vectors.
3. Sum up these vectors.
4. Put into SVM model to predict field.

## Files

* Rule.txt - contains the keywords
* candidates_of_keyword.py - the core of NLP model in charge of connecting with server/database.
* modules.py - contains reusable functions, like utilities.
* SVM.py and TFIDF.py - are class for singleton usage, only one instance in the whole system.


## Built With

* [Fasttext](https://github.com/facebookresearch/fastText) - The word vector model
* [Flashtext](https://github.com/vi3k6i5/flashtext) - Keywords extraction toolkit

## Authors

* **Chia Fang Ho** - [KellyHO](https://github.com/KellyHO)
* **Wen Bin Han** - [HanVincent](https://github.com/HanVincent)
