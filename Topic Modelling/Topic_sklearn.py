# %%
NUMBER_OF_TOPICS = 30
NUMBER_OF_TERMS_PER_TOPIC = 8
NUMBER_OF_DOCS_PER_TOPIC = 3

RUN_ALL_MODELS = False
ML_MODEL = 'nmf'
# Alternative models: 
# 'svd', 'lda'

ADDITIONAL_STOPWORDS = []

SHOW_MOST_COMMON_WORDS = False

if RUN_ALL_MODELS:
    list_models = ['nmf','svd', 'lda']
else:
    list_models = [ML_MODEL]

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME_TOPICS = "output_Topic_topics_sklearn.xlsx"
OUTPUT_FILENAME_DOCS_DISTRIB = "output_Topic_docs_distrib_sklearn.xlsx"
COLUMN_PREFIX = 'topics_sklearn'


# %%
# Brief description of the models:

# 'svd' https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis
# 'nmf' https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf
# 'lda' https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html



# %%
import os
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

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
def get_top_terms_for_topic(
    topic_terms: np.ndarray,
    feature_names: List[str],
    topic_idx: int,
    n_terms: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the top terms for a specific topic.
    
    Args:
        topic_terms: Topic-term matrix from SVD
        feature_names: List of feature names (terms)
        topic_idx: Index of the topic
        n_terms: Number of top terms to return
        
    Returns:
        List of tuples containing (term, score)
    """
    # Get the topic vector
    topic_vector = topic_terms[topic_idx]
    
    # Get indices of top terms
    top_indices = topic_vector.argsort()[-n_terms:][::-1]
    
    # Return top terms with their scores
    return [(feature_names[i], float(topic_vector[i])) for i in top_indices]

def get_top_docs_for_topic(
    topic_idx: int,
    doc_topics: np.ndarray,
    documents: List[str],
    n_docs: int = 3
):
    """
    Get the top docs for a specific topic.
    
    Args:
        doc_topics: Document-topic matrix from topic modeling
        documents: List of original document texts
        n_docs: Number of top documents to return per topic
        min_length: Minimum length of document text to consider
    """
    # Get scores for this topic across all documents
    topic_scores = doc_topics[:, topic_idx]
    
    # Get indices of top n_docs documents for this topic
    top_indices = topic_scores.argsort()[-n_docs:][::-1]
    
    # Get the corresponding documents and scores
    top_docs = []
    for idx in top_indices.tolist():
        top_docs.append((idx, topic_scores[idx].item(), documents[idx]))

    return top_docs


def get_topics_for_doc(
    doc_idx: int,
    doc_topics: np.ndarray,
    n_topics: int = 5
) -> List[Tuple[int, float]]:
    """
    Get the top topics for a specific document.
    
    Args:
        doc_idx: Index of the document
        doc_topics: Document-topic matrix 
        n_topics: Number of top topics to return
        
    Returns:
        Dictionary with model names as keys and lists of (topic_id, score) tuples as values
    """
    # Get topic scores for this document
    topics_scores = doc_topics[doc_idx]
    
    # Get top n_topics for each model
    topics = [(i+1, float(score)) for i, score in enumerate(topics_scores)]
    topics.sort(key=lambda x: x[1], reverse=True)
        
    return topics[:n_topics]




def clean_terms(top_terms):
    # Extract just the terms (without scores)
    top_terms_words = [el[0] for el in top_terms]
    
    # Keep only terms that are not substrings of others (Splitting all terms by spaces)
    word_sets = [set(s.split()) for s in top_terms_words]
    top_terms_words = [
        s for i, s in enumerate(top_terms_words)
        if not any(
            set(s.split()).issubset(other_words)
            for j, other_words in enumerate(word_sets)
            if i != j
        )
    ]
    
    return top_terms_words


def topic_results_to_df(doc_topics, topic_terms, feature_names,documents):
    # Create a DataFrame to store the results
    topics_df = pd.DataFrame(columns=[COLUMN_PREFIX+'_topic_id', COLUMN_PREFIX+'_terms', 
                                      COLUMN_PREFIX+'_term_scores', COLUMN_PREFIX+'_top_docs'])

    # Get top terms for each topic and store in DataFrame
    for topic_idx in range(NUMBER_OF_TOPICS):
        top_terms = get_top_terms_for_topic(topic_terms, feature_names, topic_idx, n_terms=NUMBER_OF_TERMS_PER_TOPIC)
        top_terms_words = clean_terms(top_terms)
        
        top_docs = get_top_docs_for_topic(topic_idx, doc_topics, documents, n_docs=NUMBER_OF_DOCS_PER_TOPIC)

        # Add to DataFrame
        topics_df = pd.concat([topics_df, pd.DataFrame({
            COLUMN_PREFIX+'_topic_id': [topic_idx + 1],
            COLUMN_PREFIX+'_terms': [top_terms_words],
            COLUMN_PREFIX+'_term_scores': [top_terms],
            COLUMN_PREFIX+'_top_docs': [top_docs]
        })], ignore_index=True)

    return topics_df

# %%
def topic_modelling_svd(
    texts: List[str],
    n_topics: int = 30,
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple = (1, 2),
    stop_words: List[str] = stopwords_list
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Perform topic modelling using SVD (Latent Semantic Analysis).
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        max_features: Maximum number of features (terms) to consider
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        ngram_rangetuple: (min_n, max_n) min_n-grams to max_n-grams will be extracted
        
    Returns:
        Tuple containing:
        - Document-topic matrix
        - Topic-term matrix
        - Feature names (terms)
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words  
    )
    
    # Transform documents to TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Perform SVD
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    doc_topics = svd.fit_transform(tfidf_matrix)
    
    # Get topic-term matrix
    topic_terms = svd.components_
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return doc_topics, topic_terms, feature_names


def topic_modelling_nmf(
    texts: List[str],
    n_topics: int = 30,
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple = (1, 2),
    stop_words: List[str] = stopwords_list,
    init: str = 'random',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Perform topic modelling using Non-negative Matrix Factorization (NMF).
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        max_features: Maximum number of features (terms) to consider
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        ngram_range: Range of n-gram sizes to consider
        stop_words: List of stop words to exclude
        init: Method for initializing the NMF algorithm ('random' or 'nndsvd')
        random_state: Random state for reproducibility
        
    Returns:
        Tuple containing:
        - Document-topic matrix
        - Topic-term matrix
        - Feature names (terms)
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words
    )
    
    # Transform documents to TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Perform NMF
    nmf = NMF(
        n_components=n_topics,
        init=init,
        random_state=random_state
    )
    doc_topics = nmf.fit_transform(tfidf_matrix)
    
    # Get topic-term matrix
    topic_terms = nmf.components_
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return doc_topics, topic_terms, feature_names


def topic_modelling_lda(
    texts: List[str],
    n_topics: int = 30,
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple = (1, 2),
    stop_words: List[str] = stopwords_list,
    learning_method: str = 'batch',
    random_state: int = 42,
    n_iter: int = 10
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Perform topic modelling using Latent Dirichlet Allocation (LDA).
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        max_features: Maximum number of features (terms) to consider
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        ngram_range: Range of n-gram sizes to consider
        stop_words: List of stop words to exclude
        learning_method: Method for updating the model ('batch' or 'online')
        random_state: Random state for reproducibility
        n_iter: Number of iterations for the learning algorithm
        
    Returns:
        Tuple containing:
        - Document-topic matrix
        - Topic-term matrix
        - Feature names (terms)
    """
    # Create CountVectorizer (LDA works with raw counts, not TF-IDF)
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words
    )
    
    # Transform documents to document-term matrix
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Perform LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method=learning_method,
        random_state=random_state,
        max_iter=n_iter
    )
    doc_topics = lda.fit_transform(doc_term_matrix)
    
    # Get topic-term matrix
    topic_terms = lda.components_
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return doc_topics, topic_terms, feature_names


def create_df_model_results(doc_topics, topic_terms, feature_names,documents,column_suffix,df):

    topics_df = topic_results_to_df(doc_topics, topic_terms, feature_names, documents)
    topics_df.sort_values(COLUMN_PREFIX+'_terms',ignore_index=True,inplace=True)
    topics_df.rename(columns={COLUMN_PREFIX+'_topic_id': COLUMN_PREFIX+'_topic_id'+column_suffix,
                              COLUMN_PREFIX+'_terms': COLUMN_PREFIX+'_terms'+column_suffix, 
                            COLUMN_PREFIX+'_term_scores': COLUMN_PREFIX+'_term_scores'+column_suffix, 
                            COLUMN_PREFIX+'_top_docs': COLUMN_PREFIX+'_top_docs'+column_suffix}, inplace=True)


    # Calculate distribution of topics per document
    docs_topic_distrib_df = df[['texts']].copy()
    docs_topic_distrib_df['idx'] = df.index
    col_docs_topic_distrib = []
    for doc_idx in df.index:
        docs_topic_distrib = get_topics_for_doc(doc_idx,doc_topics)
        col_docs_topic_distrib.append(docs_topic_distrib)
    docs_topic_distrib_df[COLUMN_PREFIX+'_docs_topic_distrib'] = col_docs_topic_distrib
    docs_topic_distrib_df.rename(columns={COLUMN_PREFIX+'_docs_topic_distrib': COLUMN_PREFIX+'_docs_topic_distrib'+column_suffix}, inplace=True)

    return topics_df,docs_topic_distrib_df

# %%
# Perform topic modelling

individual_model_results = []
individual_model_results_docs_distrib = []

if 'svd' in list_models:
    column_suffix = '_svd'
    doc_topics, topic_terms, feature_names = topic_modelling_svd(
        df['texts'].tolist(),
        n_topics=NUMBER_OF_TOPICS 
    )
    topics_df,docs_topic_distrib_df = create_df_model_results(doc_topics, topic_terms, feature_names,
                                                            df['texts'].tolist(),column_suffix,df)
    individual_model_results.append(topics_df)
    individual_model_results_docs_distrib.append(docs_topic_distrib_df)

if 'nmf' in list_models:
    column_suffix = '_nmf'
    doc_topics, topic_terms, feature_names = topic_modelling_nmf(
        df['texts'].tolist(),
        n_topics=NUMBER_OF_TOPICS 
    )
    topics_df,docs_topic_distrib_df = create_df_model_results(doc_topics, topic_terms, feature_names,
                                                            df['texts'].tolist(),column_suffix,df)
    individual_model_results.append(topics_df)
    individual_model_results_docs_distrib.append(docs_topic_distrib_df)

if 'lda' in list_models:
    column_suffix = '_lda'
    doc_topics, topic_terms, feature_names = topic_modelling_lda(
        df['texts'].tolist(),
        n_topics=NUMBER_OF_TOPICS 
    )
    topics_df,docs_topic_distrib_df = create_df_model_results(doc_topics, topic_terms, feature_names,
                                                            df['texts'].tolist(),column_suffix,df)
    individual_model_results.append(topics_df)
    individual_model_results_docs_distrib.append(docs_topic_distrib_df)

# Merge all dataframes in individual_model_results
combined_df = individual_model_results[0]
if len(individual_model_results) > 1:
    for df in individual_model_results[1:]:
        for col in df.columns:
            if col not in combined_df.columns:
                combined_df[col] = df[col]

# Merge all dataframes in individual_model_results_docs_distrib using idx as the key
combined_df_docs_distrib = individual_model_results_docs_distrib[0]
if len(individual_model_results_docs_distrib) > 1:
    for df in individual_model_results_docs_distrib[1:]:
        combined_df_docs_distrib = pd.merge(combined_df_docs_distrib, 
                                            df.drop(['texts'], axis=1), on='idx', how='outer')

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



