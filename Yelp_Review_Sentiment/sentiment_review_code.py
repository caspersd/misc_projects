#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:55:28 2024

@author: davidcaspers
"""

import pandas as pd
import regex as re
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
from sklearn.tree import plot_tree




file= 'deception_data_two_labels.csv'
df = pd.read_csv(file)

def combine_columns(column):
    combined = ''
    for value in column:
        combined += str(value).strip() + ' '
    return combined.strip()
#store labels
lie_label = df['lie']
sentiment_label = df['sentiment']

#get rid of labels in dataframe
reviews  = df.drop(columns=['lie', 'sentiment'])

#combine all review columns
reviews['reviews']= reviews.apply(combine_columns, axis=1)
reviews = reviews['reviews']

#get rid of nan and escape characters in reviews
reviews = reviews.apply(lambda x: re.sub(r'\snan\b', '', x)).apply(lambda x: re.sub(r"\\", "", x))

print("Words that were dropped:", dropwords)

#vectorize reviews, get rid of stop words
vectorizer = TfidfVectorizer(stop_words='english',norm='l2')
reviews = vectorizer.fit_transform(reviews)
column_name = vectorizer.get_feature_names_out()
reviews = pd.DataFrame(reviews.toarray(),columns=column_name)

#drop words with 2 or less characters or with more than 13 characters, remove one's that include numbers
dropwords = []
for col in reviews.columns:
    if len(col) < 3 or len(col) > 13:
        dropwords.append(col)
    if any(char.isdigit() for char in col):
        dropwords.append(col)
reviews = reviews.drop(columns=dropwords)
column_name = reviews.columns

#create dataframe with labels 
reviews_labeled = reviews.copy()
reviews_labeled['lie'] = lie_label
reviews_labeled['sentiment'] = sentiment_label

#Create word cloud for both lables

# Define the labels and their possible values
topics = {
    'lie': ['t', 'f'],           # True/False for 'lie'
    'sentiment': ['n', 'p']      # Negative/Positive for 'sentiment'
}
List_of_WC = []
# Iterate over each topic and generate word clouds for each label value
for topic, values in topics.items():
        for value in values:
            # Filter the DataFrame to get only rows for the current topic's value
            tempdf = reviews_labeled[reviews_labeled[topic] == value]
            
            # Sum the word frequencies for the topic
            word_freq = tempdf.drop(columns=['lie', 'sentiment']).sum(axis=0)
            
            # Generate a word cloud from the frequencies
            wc = WordCloud(width=1000, height=600, background_color="white",
                           min_word_length=4, max_words=200).generate_from_frequencies(word_freq)
            
            # Store the word cloud in the list
            List_of_WC.append((topic, value, wc))

# Plotting the word clouds
for topic, value, wc in List_of_WC:
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.title(f'Word Cloud for {topic} = {value}')
    plt.show()



#plot most frequent words
word_count = reviews.sum(axis=0)
word_frequency_df = pd.DataFrame({'word': column_name, 'count': word_count})
most_words = word_frequency_df.sort_values(by='count',ascending=False).head(20)

fig = plt.figure(figsize=(10, 5))
plt.bar(most_words['word'], most_words['count'], color='maroon', width=0.4)
plt.xlabel('Words')
plt.ylabel('TF-IDF Frequency')
plt.title('Most Common TF-IDF Weighted Words in Reviews')
plt.xticks(rotation=45)
plt.show()

# Boxplot of the distribution of words
fig2 = plt.figure(figsize=(10, 5))
plt.boxplot(word_frequency_df['count'])
plt.ylabel('TD-IDF Frequency of Word')
plt.title('Distribution of Word Count in Reviews')
plt.show()



#create function to evaluate models


def evaluate_model(x, y):
    models = {
        'MNB': MultinomialNB(),
        'BNB': BernoulliNB(), #binarize attribute (default) will convert counts to binary automatically
        'DT': DecisionTreeClassifier()
    }
    
    kf = KFold(n_splits=10)
    results = {}
    
    for name, model in models.items():
        y_pred = cross_val_predict(model, x, y, cv=kf)
        precision, recall, fscore, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        accuracy = accuracy_score(y, y_pred)
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'fitted_model': model.fit(x, y)
        }
    
    return results

#create function to evaluate top indicitive words for MNM and BNB:
def show_top20(model,column_names,label):
    for i, category in enumerate(label.unique()):
        top20 = np.argsort(model.feature_log_prob_[i])[-20:]
        print("%s: %s" % (category, " ".join(column_names[top20])))
        
#create function to evaluate top indicitive words for MNM and BNB:
def show_top20_DT(model,column_names):
    top20 = np.argsort(model.feature_importances_)[-20:]
    print("Highly Discriminative Words: %s" % (" ".join(column_names[top20])))


# Evaluate models for lie
sentiment_results = evaluate_model(reviews, sentiment_label)
print(f"\n BNB Sentiment Model - \n  Precision: {sentiment_results['BNB']['precision']} \n  Recall: {sentiment_results['BNB']['recall']} \n  Accuracy: {sentiment_results['BNB']['accuracy']} \n  F1: {sentiment_results['BNB']['fscore']}")
print(f"MNB Sentiment Model - \n  Precision: {sentiment_results['MNB']['precision']} \n  Recall: {sentiment_results['MNB']['recall']} \n  Accuracy: {sentiment_results['MNB']['accuracy']} \n  F1: {sentiment_results['MNB']['fscore']}")
print(f"DT Sentiment Model - \n  Precision: {sentiment_results['DT']['precision']} \n  Recall: {sentiment_results['DT']['recall']} \n  Accuracy: {sentiment_results['DT']['accuracy']} \n  F1: {sentiment_results['DT']['fscore']}")

#show top 20
print("\n Sentiment Label, Significant Words for BNB Model:")
show_top20(sentiment_results['BNB']['fitted_model'],column_name,sentiment_label)
print("\n Sentiment Label, Significant Words for MNB Model:")
show_top20(sentiment_results['MNB']['fitted_model'],column_name,sentiment_label)
print("\n Sentiment Label, Significant Words for DT Model:")
show_top20_DT(sentiment_results['DT']['fitted_model'],column_name)
# Plotting the Decision Tree for Sentiment Model
dt_model_sentiment = sentiment_results['DT']['fitted_model']
plt.figure(figsize=(20, 10))  
plot_tree(dt_model_sentiment, filled=True, feature_names=column_name, class_names=sentiment_label.unique(), rounded=True)
plt.title("Decision Tree for Sentiment Classification")
plt.show()


# Evaluate models for lie detection
lie_results = evaluate_model(reviews, lie_label)
print(f"\n BNB Lie Model - \n  Precision: {lie_results['BNB']['precision']} \n  Recall: {lie_results['BNB']['recall']} \n  Accuracy: {lie_results['BNB']['accuracy']} \n  F1: {lie_results['BNB']['fscore']}")
print(f"MNB Lie Model - \n  Precision: {lie_results['MNB']['precision']} \n  Recall: {lie_results['MNB']['recall']} \n  Accuracy: {lie_results['MNB']['accuracy']} \n  F1: {lie_results['MNB']['fscore']}")
print(f"DT Lie Model - \n  Precision: {lie_results['DT']['precision']} \n  Recall: {lie_results['DT']['recall']} \n  Accuracy: {lie_results['DT']['accuracy']} \n  F1: {lie_results['DT']['fscore']}")

print("\n Lie Label, Significant Words for BNB Model:")
show_top20(lie_results['BNB']['fitted_model'],column_name,lie_label)
print("\n Lie Label, Significant Words for MNB Model:")
show_top20(lie_results['MNB']['fitted_model'],column_name,lie_label)
print("\n Lie Label, Significant Words for DT Model:")
show_top20_DT(lie_results['DT']['fitted_model'],column_name)

# Plotting the Decision Tree for Lie Detection Model
dt_model_lie = lie_results['DT']['fitted_model']
plt.figure(figsize=(20, 10))  
plot_tree(dt_model_lie, filled=True, feature_names=column_name, class_names=lie_label.unique(), rounded=True)
plt.title("Decision Tree for Lie Detection")
plt.show()


