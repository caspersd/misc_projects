import requests  #to query the API 
import re  #regular expressions
import pandas as pd   # for dataframes
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

End="https://newsapi.org/v2/everything"
URLPost = {'apiKey': '4263d703533c4b4792cbcd114cd996b9',
           'sortBy':'popularity',
           'q': 'Walz',
           'from': 2024-8-6,
           'language': 'en'
       }


request1 = requests.get(End, URLPost)
jsontxt1= request1.json()
articles = jsontxt1.get('articles',[])

df1 = pd.DataFrame(articles)
df1['Label'] = 'walz'


End="https://newsapi.org/v2/everything"
URLPost = {'apiKey': '4263d703533c4b4792cbcd114cd996b9',
           'sortBy':'popularity',
           'q': 'Vance',
           'from': 2024-8-6,
           'language': 'en'
       }


request2 = requests.get(End, URLPost)
jsontxt2= request2.json()
articles = jsontxt2.get('articles',[])
df2 = pd.DataFrame(articles)
df2['Label'] = 'vance'


End="https://newsapi.org/v2/everything"
URLPost = {'apiKey': '4263d703533c4b4792cbcd114cd996b9',
           'sortBy':'popularity',
           'q': 'Trump',
           'from': 2024-8-6,
           'language': 'en'
       }


request3 = requests.get(End, URLPost)
jsontxt3= request3.json()
articles = jsontxt3.get('articles',[])
df3 = pd.DataFrame(articles)
df3['Label'] = 'trump'


End="https://newsapi.org/v2/everything"
URLPost = {'apiKey': '4263d703533c4b4792cbcd114cd996b9',
           'sortBy':'popularity',
           'q': 'Trump',
           'from': 2024-8-6,
           'language': 'en'
       }


request4 = requests.get(End, URLPost)
jsontxt4= request4.json()
articles = jsontxt4.get('articles',[])
df4 = pd.DataFrame(articles)
df4['Label'] = 'kamala'


df = pd.concat([df1, df2, df3, df4],ignore_index=True)
#drop irrelevant columns
df = df[['publishedAt', 'title', 'description', 'content', 'source','Label']]


#Get rid of id in source column and keep only name
df['source'] = df['source'].apply(lambda x: x['name'])
#clean up date column
df['publishedAt'] = df['publishedAt'].apply(lambda x: x.split("T")[0])
df.rename(columns={'publishedAt':'Date'},inplace=True)

#clean up title
df['title'] = df['title'].apply(lambda x: re.sub(r'[,.;@#?!&$\-\']+', ' ', x, flags=re.IGNORECASE))
df['title'] = df['title'].apply(lambda x: re.sub(' +', ' ', x, flags=re.IGNORECASE))
df['title'] = df['title'].apply(lambda x: re.sub(r'\"', ' ', x, flags=re.IGNORECASE))
df['title'] = df['title'].apply(lambda x: re.sub(r'[^a-zA-Z]', " ", x, flags=re.VERBOSE))
df['title'] = df['title'].apply(lambda x: re.sub(',', '',x))
df['title'] = df['title'].apply(lambda x: ' '.join(x.split()))
df['title'] = df['title'].apply(lambda x: re.sub("\n|\r", "", x))

#clean up description, relabeled as headline
df.rename(columns={'description':'headline'},inplace=True)
df['headline'] = df['headline'].apply(lambda x: re.sub(r'[,.;@#?!&$\-\']+', ' ', x, flags=re.IGNORECASE))
df['headline'] = df['headline'].apply(lambda x: re.sub(' +', ' ', x, flags=re.IGNORECASE))
df['headline'] = df['headline'].apply(lambda x: re.sub(r'\"', ' ', x, flags=re.IGNORECASE))
df['headline'] = df['headline'].apply(lambda x: re.sub(r'[^a-zA-Z]', " ", x, flags=re.VERBOSE))
df['headline'] = df['headline'].apply(lambda x: re.sub(',', '',x))
df['headline'] = df['headline'].apply(lambda x: ' '.join(x.split()))
df['headline'] = df['headline'].apply(lambda x: re.sub("\n|\r", "", x))

#clean up content
df['content'] = df['content'].apply(lambda x: re.sub(r'[,.;@#?!&$\-\']+', ' ', x, flags=re.IGNORECASE))
df['content'] = df['content'].apply(lambda x: re.sub(' +', ' ', x, flags=re.IGNORECASE))
df['content'] = df['content'].apply(lambda x: re.sub(r'\"', ' ', x, flags=re.IGNORECASE))
df['content'] = df['content'].apply(lambda x: re.sub(r'[^a-zA-Z]', " ", x, flags=re.VERBOSE))
df['content'] = df['content'].apply(lambda x: re.sub(',', '',x))
df['content'] = df['content'].apply(lambda x: ' '.join(x.split()))
df['content'] = df['content'].apply(lambda x: re.sub("\n|\r", "", x))

#remove rows with NaN in them
df = df.dropna()

headlines = df['headline']

#get a list of labels and ensure they're lower case
topics = list(set(df['Label']))
topics = [topic.lower() for topic in topics]
remove_words = topics.copy()
remove_words.extend(['harris', 'donald', 'jd','tim'])

new_headlines = []
#remove words that are the same as our labels from the headlines
for headline in headlines:
    words = headline.split(" ")
    filtered_words = []
    for word in words:
        word=word.lower()
        if word not in remove_words:
            filtered_words.append(word)
    filtered_headline=" ".join(filtered_words)
    new_headlines.append(filtered_headline)
headlines = new_headlines

#creates a dataframe with just labels
labels = df['Label']

#instantiate count vectorizer
MyCountV = CountVectorizer(input="content",lowercase=True, stop_words="english", max_features=50
    )

tokens = MyCountV.fit_transform(headlines)
columns = MyCountV.get_feature_names_out()
#create dataframe of word tokens from headlines
Headlines_DF = pd.DataFrame(tokens.toarray(),columns=columns)


#save dataframe in case we need to revert to the original
Orig_DF = Headlines_DF.copy()

labeled_Headlines_DF = pd.concat([labels,Headlines_DF],axis=1,join='inner')

###########################################################
#
#
#                   Create Word Clouds
#
#
###########################################################

# Create a list to store word cloud objects
List_of_WC = []

for mytopic in topics:
    # Filter the DataFrame to get only rows for the current topic
    tempdf = labeled_Headlines_DF[labeled_Headlines_DF['Label'] == mytopic]
    
    # Sum the word frequencies for the topic
    tempdf = tempdf.drop(columns=['Label']).sum(axis=0)
    
    # Generate a word cloud from the frequencies
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                            min_word_length=4, 
                            max_words=200).generate_from_frequencies(tempdf)
    
    # Store the word cloud in the list
    List_of_WC.append(NextVarName)

# Plotting the word clouds
fig = plt.figure(figsize=(25, 25))
NumTopics = len(topics)
for i in range(NumTopics):
    ax = fig.add_subplot(NumTopics, 1, i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.title(f"Word Cloud for {topics[i].capitalize()}", fontsize=20)
    plt.axis("off")

# Save the word clouds to a file
plt.savefig("NewClouds.pdf")
plt.show()

    

###########################################################
#
#
#                   Clustering
#
#
###########################################################


inertia_values = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Headlines_DF)
    inertia_values.append(kmeans.inertia_)

# Plotting the elbow plot
plt.figure(figsize=(10, 7))
plt.plot(K, inertia_values, 'bo-', marker='o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('Elbow Method For Optimal k')
plt.xticks(K)
plt.grid(True)
plt.show()

 # KMeans Clustering
My_KMean = KMeans(n_clusters=4)
My_KMean.fit(Headlines_DF)
My_labels = My_KMean.predict(Headlines_DF)
print("KMeans Labels (4 clusters):\n", My_labels)

kmeans_df = Headlines_DF.copy()
kmeans_df['Cluster'] = My_labels
kmeans_df['Label'] = labels

cluster_counts = kmeans_df.groupby(['Cluster', 'Label']).size().unstack()

# Plotting the count of articles in each cluster, broken down by label
plt.figure(figsize=(10, 7))
cluster_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10,7))
plt.title('Total Count of Articles in Each Cluster by Label')
plt.xlabel('Cluster')
plt.ylabel('Number of Articles')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


# Cosine Similarity and Hierarchical Clustering
cosdist = 1 - cosine_similarity(Headlines_DF)
print("Cosine Distances:\n", np.round(cosdist, 3))

# Hierarchical Clustering using ward and cosine similarity
linkage_matrix = ward(cosdist)
print("Linkage Matrix:\n", linkage_matrix)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(linkage_matrix)
plt.show()

cluster_labels = fcluster(linkage_matrix, 4, criterion='maxclust')
kmeans_df['Hierarchical_Cluster'] = cluster_labels

cluster_counts = kmeans_df.groupby(['Hierarchical_Cluster', 'Label']).size().unstack()

# Plotting the count of articles in each cluster, broken down by label
plt.figure(figsize=(10, 7))
cluster_counts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10,7))
plt.title('Total Count of Articles in Each Cluster by Label')
plt.xlabel('Cluster')
plt.ylabel('Number of Articles')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

###########################################################
#
#
#                   Split Data into Training / Test
#
#
###########################################################

TrainDF, TestDF = train_test_split(labeled_Headlines_DF, test_size=0.3)

# Separate labels
TrainLabels = TrainDF['Label']
TestLabels = TestDF['Label']

# Remove labels from the data
TrainDF = TrainDF.drop(columns=['Label'])
TestDF = TestDF.drop(columns=['Label'])

##################################################
# Multinomial Naive Bayes
##################################################

# Instantiate and fit the model
MyModelNB = MultinomialNB()
MyNB = MyModelNB.fit(TrainDF, TrainLabels)

# Predict and calculate probabilities
Prediction = MyModelNB.predict(TestDF)
print(MyModelNB.classes_)
print("Naive Bayes Prediction Probabilities:\n", MyModelNB.classes_,"\n",np.round(MyModelNB.predict_proba(TestDF), 2))

# Confusion Matrix
cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("\nNaive Bayes Confusion Matrix:\n", cnf_matrix)


# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=topics, yticklabels=topics)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Multinomial Naive Bayes Confusion Matrix')
plt.show()

##################################################
# Decision Tree
##################################################

# Instantiate and fit the Decision Tree model
MyDT = DecisionTreeClassifier(criterion='entropy')
MyDT.fit(TrainDF, TrainLabels)

# Visualize the decision tree
feature_names = TrainDF.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                                   feature_names=feature_names,
                                   class_names=topics,
                                   filled=True, rounded=True,
                                   special_characters=True)
graph = graphviz.Source(Tree_Object)
graph.render("MyTree")

# Decision Tree Predictions and Confusion Matrix
DT_pred = MyDT.predict(TestDF)
bn_matrix = confusion_matrix(TestLabels, DT_pred)
# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(bn_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=topics, yticklabels=topics)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Decision Tree Confusion Matrix')
plt.show()



# Feature Importance
FeatureImp = MyDT.feature_importances_
indices = np.argsort(FeatureImp)[::-1]

# Print out the important features
print("\nImportant Features in Decision Tree:")
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print(f"{f + 1}. Feature {indices[f]} ({FeatureImp[indices[f]]}) - {feature_names[indices[f]]}")