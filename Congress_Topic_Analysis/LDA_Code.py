# -*- coding: utf-8 


import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import contractions


# Download the necessary NLTK datasets
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')



# Read corpus into a dataframe
base_path = 'lda_wordcloud_results'
rep_corpus_path = os.path.join(base_path, 'republican')
dem_corpus_path = os.path.join(base_path, 'democrat')
fr_corpus_path = os.path.join(rep_corpus_path, 'female')
mr_corpus_path = os.path.join(rep_corpus_path, 'male')
fd_corpus_path = os.path.join(dem_corpus_path, 'female')
md_corpus_path = os.path.join(dem_corpus_path, 'male')

def read_corpus_into_dataframe(corpus_path, gender, party):
    transcripts = []
    for filename in os.listdir(corpus_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(corpus_path, filename)
            with open(filepath, 'r', encoding='utf-8',errors='ignore') as file:
                transcript = file.read()
                transcripts.append({'filename': filename, 'transcript': transcript, 'party': party, 'gender': gender})
    return pd.DataFrame(transcripts)

# Correctly use the path variables for each gender and party combination
fr_transcript = read_corpus_into_dataframe(fr_corpus_path, 'female', 'republican')
mr_transcript = read_corpus_into_dataframe(mr_corpus_path, 'male', 'republican')
fd_transcript = read_corpus_into_dataframe(fd_corpus_path, 'female', 'democrat')
md_transcript = read_corpus_into_dataframe(md_corpus_path, 'male', 'democrat')

# Concatenate all dataframes
df = pd.concat([fr_transcript, mr_transcript, fd_transcript, md_transcript], ignore_index=True)

#split labels and reviews
df.rename(columns={'party': 'party_label', 'gender': 'gender_label'}, inplace=True)
labels = df[['party_label', 'gender_label']]
transcripts = df['transcript']
file_names = df['filename']


# Regular expression pattern to extract text between <TEXT> and </TEXT>
pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.DOTALL)
# Use a lambda function to extract text between <TEXT> and </TEXT> tags
transcripts = transcripts.apply(lambda x: ' '.join(pattern.findall(x)))

transcripts = transcripts.apply(lambda x: contractions.fix(x))


#clean transcripts
transcripts = transcripts.str.replace(r'\b\w*\d+\w*\b', '', regex=True) #get rid of numerals and words with numerals in them
transcripts = transcripts.str.replace(r'[^a-zA-Z\s]',' ', regex=True) #get rid of punctuation and other non alphabetical characters
transcripts = transcripts.str.replace(r'\s+', ' ', regex=True)  # Replaces multiple spaces with a single space
transcripts = transcripts.str.strip()  # Removes leading and trailing spaces

# Lemmatize cleaned text
def get_wordnet_pos(treebank_tag):
    """
    Convert Treebank POS tag to WordNet POS tag for lemmatization.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    """
    Lemmatize the input text using WordNetLemmatizer and POS tags.
    """
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)  # Tokenize the text
    pos_tags = pos_tag(words)  # Get POS tags
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]  # Lemmatize with POS tags
    return ' '.join(lemmatized_words)

# Apply lemmatization to the entire column of transcripts
lemmatized_transcripts = transcripts.apply(lemmatize_text)

#Do remaining cleaning steps
lemmatized_transcripts= lemmatized_transcripts.str.replace(r'\b[A-Z]+\b', '',regex=True)  #remove speaker's names
lemmatized_transcripts = lemmatized_transcripts.str.replace(r'\b(Mr|Mrs|Ms)\b','',regex=True) #remove Mr, Mrs, Ms
lemmatized_transcripts = lemmatized_transcripts.str.replace(r'\s+', ' ', regex=True)  # Replaces multiple spaces with a single space
lemmatized_transcripts = lemmatized_transcripts.str.strip()  # Removes leading and trailing spaces


#Create count dataframe
MyCountV = CountVectorizer(stop_words='english')
tokens = MyCountV.fit_transform(lemmatized_transcripts)
columns = MyCountV.get_feature_names_out()
transcripts_vectorized = pd.DataFrame(tokens.toarray(), columns=columns)


#create new dataframe with labels
transcripts_labeled = pd.concat([transcripts_vectorized, labels.reset_index(drop=True)], axis=1)

# Define the labels and their possible values
parties = ['republican', 'democrat']
genders = ['female', 'male']
List_of_WC = []

# Iterate over each combination of party and gender
for party in parties:
    for gender in genders:
        # Filter the DataFrame to get only rows for the current party and gender
        tempdf = transcripts_labeled[(transcripts_labeled['party_label'] == party) & (transcripts_labeled['gender_label'] == gender)]
        
        # Sum the word frequencies for the filtered DataFrame
        word_freq = tempdf.drop(columns=['party_label', 'gender_label']).sum(axis=0)
        
        # Generate a word cloud from the frequencies
        wc = WordCloud(width=1000, height=600, background_color="white",
                       min_word_length=4, max_words=200).generate_from_frequencies(word_freq)
        
        # Store the word cloud in the list
        List_of_WC.append((party, gender, wc))

# Plotting the word clouds
for party, gender, wc in List_of_WC:
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.title(f'Word Cloud for {party} {gender}')
    plt.show()
    
    

# Define a list of topic numbers to run LDA for
topic_numbers = [5, 10, 20, 30]

# Loop through each topic number
for n_topics in topic_numbers:
    print(f"\nRunning LDA for {n_topics} topics...\n")

    # Define LDA model with the current number of topics
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,  # Number of topics
        max_iter=10,  # Maximum number of iterations
        random_state=42,
        n_jobs=-1,  # Use all available CPUs
        learning_method='batch'  # Using 'batch' for full-batch updates
    )

    # Fit LDA model
    lda_model.fit(transcripts_vectorized)

    # Function to display topics and their associated words
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx + 1}:")
            print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    # Display topics
    no_top_words = 10
    display_topics(lda_model, columns, no_top_words)

    # Visualize the topics using word clouds
    for topic_idx, topic in enumerate(lda_model.components_):
        plt.figure()
        plt.imshow(WordCloud(background_color='white').fit_words(dict(zip(columns, topic))))
        plt.axis("off")
        plt.title(f"Topic {topic_idx + 1} for {n_topics} topics")
        plt.show()





# Function to perform LDA analysis and print topics
def lda_analysis(df, num_topics, title_prefix):
    # Filter out label columns
    tokens = df.drop(columns=['party_label', 'gender_label'])
    
    lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, random_state=42, learning_method='batch')
    lda_model.fit(tokens)
    
    feature_names = tokens.columns

    # Display topics and their associated words
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"{title_prefix} - Topic {topic_idx + 1}: {', '.join([feature_names[i] for i in topic.argsort()[:-11:-1]])}")

# Perform LDA analysis on Female subset
female_subset = transcripts_labeled[transcripts_labeled['gender_label'] == 'female']
print("\nLDA with 20 topics for Female")
lda_analysis(female_subset, num_topics=20, title_prefix='Female')

# Perform LDA analysis on Male subset
male_subset = transcripts_labeled[transcripts_labeled['gender_label'] == 'male']
print("\nLDA with 20 topics for Male")
lda_analysis(male_subset, num_topics=20, title_prefix='Male')

# Perform LDA analysis on Democrat subset
democrat_subset = transcripts_labeled[transcripts_labeled['party_label'] == 'democrat']
print("\nLDA with 20 topics for Democrat")
lda_analysis(democrat_subset, num_topics=20, title_prefix='Democrat')

# Perform LDA analysis on Republican subset
republican_subset = transcripts_labeled[transcripts_labeled['party_label'] == 'republican']
print("\nLDA with 20 topics for Republican")
lda_analysis(republican_subset, num_topics=20, title_prefix='Republican')