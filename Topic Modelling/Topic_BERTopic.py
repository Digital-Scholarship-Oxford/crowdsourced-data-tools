# %%
NUMBER_OF_TOPICS = 30
NUMBER_OF_TERMS_PER_TOPIC = 8
NUMBER_OF_DOCS_PER_TOPIC = 3

RUN_ALL_MODELS = False
ML_MODEL = "all-MiniLM-L6-v2"
# Alternative models: 
# ML_MODEL = "BAAI/bge-m3"
# ML_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# ML_MODEL = "ibm-granite/granite-embedding-125m-english"
# ML_MODEL = "ibm-granite/granite-embedding-278m-multilingual"

ADDITIONAL_STOPWORDS = []

SHOW_MOST_COMMON_WORDS = False

if RUN_ALL_MODELS:
    MODEL_LIST = ['BAAI/bge-m3','all-MiniLM-L6-v2', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
              'ibm-granite/granite-embedding-125m-english', 'ibm-granite/granite-embedding-278m-multilingual']
else:
    MODEL_LIST = [ML_MODEL]

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME_TOPICS = "output_Topic_topics_BERTopic.xlsx"
OUTPUT_FILENAME_DOCS_DISTRIB = "output_Topic_docs_distrib_BERTopic.xlsx"
COLUMN_PREFIX = 'topics_BERTopic'


# %%
# Brief description of the models:

# https://maartengr.github.io/BERTopic/index.html
# https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html#sentence-transformers


# %%
import os
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
from bertopic.dimensionality import BaseDimensionalityReduction


folder_file_path = "./"

file_path = os.path.join(folder_file_path, FILENAME)
output_file_path = os.path.join(folder_file_path, OUTPUT_FILENAME_TOPICS)
output_file_path_docs_distrib = os.path.join(folder_file_path, OUTPUT_FILENAME_DOCS_DISTRIB)

# Load the Excel file into a pandas DataFrame
try:
    df = pd.read_excel(file_path,header=None)

    # Check if first row contains 'texts' and convert it to header if so, else add header
    if len(df) > 0 and any('texts'==str(cell) if pd.notna(cell) else False for cell in df.iloc[0]):
        # Use first row as header
        df.columns = df.iloc[0]
        # Remove the first row from data
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = ['texts']
        df = df.reset_index(drop=True)    
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except ValueError:
    print("ValueError: The input file should be a single column of texts.")
    exit()

# %%
df

# %%
# In BERTopic, stopwords and the Vectorizer are used after the representation clustering. 
# Thus, they don't affect the embeddings

def get_additional_tokenized_stop_words_vectorizer(stopwords_list):
    vectorizer = CountVectorizer(
        stop_words=stopwords_list,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 1),
        token_pattern=f'[a-zA-Z]{{{2},}}' 
    )

    analyzer = vectorizer.build_analyzer()
    additional_tokenized_stop_words = analyzer(' '.join(stopwords_list))
    
    return additional_tokenized_stop_words

stopwords_list = list(en_stop)
stopwords_list = stopwords_list + get_additional_tokenized_stop_words_vectorizer(stopwords_list)
stopwords_list = stopwords_list + ADDITIONAL_STOPWORDS

# %%
def get_most_common_words(
    text: str,
    n_words: int = 100,
    stop_words: List[str] = stopwords_list,
    min_word_length: int = 2,
    ngram_range: tuple = (1, 2)
) -> List[Tuple[str, int]]:
    """
    Get the most common words in a text string.
    
    Args:
        text: Input text string
        n_words: Number of top words to return
        stop_words: List of stop words to exclude
        min_word_length: Minimum length of words to consider
        ngram_range: Range of n-gram sizes to consider
        
    Returns:
        List of tuples containing (word, count)
    """
    # Create CountVectorizer
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        min_df=1,
        max_df=1.0,
        ngram_range=ngram_range,
        token_pattern=f'[a-zA-Z]{{{min_word_length},}}' 
    )
    
    # Fit and transform the text
    counts = vectorizer.fit_transform([text])
    
    # Get feature names and counts
    feature_names = vectorizer.get_feature_names_out()
    word_counts = counts.toarray()[0]
    
    # Create list of (word, count) tuples and sort by count
    word_count_pairs = list(zip(feature_names, word_counts))
    word_count_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top n_words
    return word_count_pairs[:n_words]

if SHOW_MOST_COMMON_WORDS:
    full_text = ' '.join(df['texts'])
    common_words = get_most_common_words(full_text)
    print('Most common words in the text (may be selected as stopwords):\n')
    for word, count in common_words:
        print(f"{word}: {count}")

# %%
def get_top_docs_for_topic(
    topic_idx: int,
    topic_model: BERTopic,
    documents: List[str]
):
    temp_df=topic_model.get_document_info(documents)
    top_docs_list = []
    if 'HDBSCAN' in str(topic_model.get_params()['hdbscan_model']):
        for idx in temp_df[(temp_df['Representative_document']==True)&(temp_df['Topic']==topic_idx)].index:
            top_docs_list.append((idx,temp_df.loc[idx,'Probability'].item(),temp_df.loc[idx,'Document']))
    elif 'KMeans' in str(topic_model.get_params()['hdbscan_model']):
        for idx in temp_df[(temp_df['Representative_document']==True)&(temp_df['Topic']==topic_idx)].index:
            top_docs_list.append((idx,None,temp_df.loc[idx,'Document']))
    else:
        assert False, "Not implemented"
    top_docs_list

    return top_docs_list

# %%
def topic_modelling_bertopic(
    texts: List[str],
    n_topics: int = 30,
    top_n_words: int = 10,
    min_topic_size: int = 5,
    n_gram_range: tuple = (1, 2),
    stop_words: List[str] = stopwords_list,
    embedding_model_str: str = "all-MiniLM-L6-v2" 
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Perform topic modelling using BERTopic.
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        min_topic_size: Minimum size of a topic
        n_gram_range: Range of n-gram sizes to consider
        stop_words: List of stop words to exclude
        embedding_model: Name of the sentence transformer model to use
        
    Returns:
        Tuple containing:
        - Document-topic matrix
        - Topic-term matrix
        - Feature names (terms)
    """
    
    # Create vectorizer for custom stop words 
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=n_gram_range,
        min_df=1,
        max_df=1.0,
    )
    
    embedding_model = SentenceTransformer(embedding_model_str)
    # embedding_model = pipeline("feature-extraction", model="distilbert-base-cased")

    # # BERTopic - HDBSCAN - No dimensionality reduction
    # empty_dimensionality_model = BaseDimensionalityReduction()
    # topic_model = BERTopic(
    # nr_topics=n_topics,
    # top_n_words=top_n_words,
    # min_topic_size=min_topic_size,
    # vectorizer_model=None,
    # embedding_model=embedding_model,
    # umap_model=empty_dimensionality_model
    # )
    
    # # BERTopic - HDBSCAN - Dimensionality reduction
    # from umap import UMAP
    # umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine')
    # topic_model = BERTopic(
    # nr_topics=n_topics,
    # top_n_words=top_n_words,
    # min_topic_size=min_topic_size,
    # vectorizer_model=None,
    # embedding_model=embedding_model,
    # umap_model=umap_model
    # )
    
    # # BERTopic - KMeans - No dimensionality reduction
    # empty_dimensionality_model = BaseDimensionalityReduction()
    # from sklearn.cluster import KMeans
    # cluster_model = KMeans(n_clusters=n_topics)
    # topic_model = BERTopic(
    #     nr_topics=n_topics,
    #     top_n_words=top_n_words,
    #     min_topic_size=min_topic_size,
    #     vectorizer_model=None,
    #     embedding_model=embedding_model,
    #     hdbscan_model=cluster_model,
    #     umap_model=empty_dimensionality_model
    # )

    # BERTopic - KMeans - Dimensionality reduction
    from sklearn.cluster import KMeans
    cluster_model = KMeans(n_clusters=n_topics)
    topic_model = BERTopic(
        nr_topics=n_topics,
        top_n_words=top_n_words,
        min_topic_size=min_topic_size,
        vectorizer_model=None,
        embedding_model=embedding_model,
        hdbscan_model=cluster_model
    )

    # Fit and transform the documents
    topics, probs = topic_model.fit_transform(texts)    
    topic_model.update_topics(texts, vectorizer_model=vectorizer_model,
                              n_gram_range=n_gram_range, top_n_words=NUMBER_OF_TERMS_PER_TOPIC)

    return topics, probs, topic_model

# %%
individual_model_results = []
individual_model_results_docs_distrib = []

for EMBEDDING_MODEL in MODEL_LIST:

    print(f"Performing topic modelling with model: {EMBEDDING_MODEL}")

    topics, probs, topic_model = topic_modelling_bertopic(
        df['texts'].tolist(),
        n_topics=NUMBER_OF_TOPICS,
        top_n_words=NUMBER_OF_TERMS_PER_TOPIC,
        embedding_model_str=EMBEDDING_MODEL
    )

    topics_df = topic_model.get_topic_info()
    topics_df.drop(['Name','Representative_Docs'], axis=1, inplace=True)
    topics_df.rename(columns={'Topic': COLUMN_PREFIX+'_topic_id_'+EMBEDDING_MODEL, 'Count': COLUMN_PREFIX+'_docs_count_'+EMBEDDING_MODEL, 
                              'Representation': COLUMN_PREFIX+'_terms_'+EMBEDDING_MODEL}, inplace=True)    
    topics_df[COLUMN_PREFIX+'_top_docs_'+EMBEDDING_MODEL] = topics_df[COLUMN_PREFIX+'_topic_id_'+EMBEDDING_MODEL].apply(
        lambda x: get_top_docs_for_topic(x,topic_model,df['texts'].tolist()))
    topics_df[COLUMN_PREFIX+'_terms_scores_'+EMBEDDING_MODEL] = topics_df[COLUMN_PREFIX+'_topic_id_'+EMBEDDING_MODEL].apply(lambda x: topic_model.get_topics()[x])
    topics_df[COLUMN_PREFIX+'_terms_scores_'+EMBEDDING_MODEL] = topics_df[COLUMN_PREFIX+'_terms_scores_'+EMBEDDING_MODEL].apply(lambda x: [(x[0],float(x[1])) for x in x])
    topics_df.sort_values(COLUMN_PREFIX+'_terms_'+EMBEDDING_MODEL,ignore_index=True,inplace=True)
    individual_model_results.append(topics_df)


    # Calculate distribution of topics per document
    docs_topic_distrib_df = df[['texts']].copy()
    docs_topic_distrib_df['idx'] = df.index
    topic_distr, topic_token_distr = topic_model.approximate_distribution(df['texts'].tolist())
    col_docs_topic_distrib = []
    for doc_idx in df.index:
        docs_topic_distrib = []
        for topic_idx,topic_percent in enumerate(topic_distr[doc_idx,:].tolist()):
            if topic_percent != 0:
                docs_topic_distrib.append((topic_idx,topic_percent))
        col_docs_topic_distrib.append(docs_topic_distrib)
    docs_topic_distrib_df[COLUMN_PREFIX+'_docs_topic_distrib_'] = col_docs_topic_distrib
    docs_topic_distrib_df.rename(columns={COLUMN_PREFIX+'_docs_topic_distrib_': COLUMN_PREFIX+'_docs_topic_distrib_'+EMBEDDING_MODEL}, inplace=True)
    individual_model_results_docs_distrib.append(docs_topic_distrib_df)

if RUN_ALL_MODELS:
    # Merge all dataframes in individual_model_results
    combined_df = individual_model_results[0]
    for df in individual_model_results[1:]:
        for col in df.columns:
            if col not in combined_df.columns:
                combined_df[col] = df[col]
    # Merge all dataframes in individual_model_results_docs_distrib using idx as the key
    combined_df_docs_distrib = individual_model_results_docs_distrib[0]
    for df in individual_model_results_docs_distrib[1:]:
        combined_df_docs_distrib = pd.merge(combined_df_docs_distrib, 
                                            df.drop(['texts'], axis=1), on='idx', how='outer')
else:
    combined_df = topics_df
    combined_df_docs_distrib = docs_topic_distrib_df


# %%
combined_df

# %%
combined_df_docs_distrib

# %%
def save_df_to_excel(df, file_path):
    # Save the DataFrame to Excel
    df.to_excel(file_path, index=False)
    
    # Load the file back to verify
    df_load = pd.read_excel(file_path)
    
    # Verify the DataFrames are equal
    try:       
        # Convert both DataFrames to string for comparison (handles dict/set serialization)
        df_str = df.astype(str)
        df_load_str = df_load.astype(str)        
        # Check if content is equal
        if df_str.equals(df_load_str):
            return True
        else:
            return False            
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False

try:
    # SAVE the output file
    success = save_df_to_excel(combined_df, output_file_path)
    if success:
        print(f"\nFile successfully saved and verified at: {output_file_path}")
    else:
        print(f"\nFile saved but verification failed. Please check the output manually.")
except Exception as e:
    print(f"Error saving the file: {e}")

try:
    # SAVE the output file
    success = save_df_to_excel(combined_df_docs_distrib, output_file_path_docs_distrib)
    if success:
        print(f"\nFile successfully saved and verified at: {output_file_path_docs_distrib}")
    else:
        print(f"\nFile saved but verification failed. Please check the output manually.")
except Exception as e:
    print(f"Error saving the file: {e}")

# %%



