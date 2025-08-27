# %%
NUMBER_OF_KEYWORDS = 5

RUN_ALL_EXTRACTORS = False
EXTRACTOR_SELECTED = 'tfidf'
# EXTRACTOR_SELECTED = 'kpminer'
# EXTRACTOR_SELECTED = 'yake'
# EXTRACTOR_SELECTED = 'textrank'
# EXTRACTOR_SELECTED = 'singlerank'
# EXTRACTOR_SELECTED = 'topicrank'
# EXTRACTOR_SELECTED = 'positionrank'
# EXTRACTOR_SELECTED = 'multipartiterank'

JOIN_DOCUMENTS_BEFORE_EXTRACTION = False
NUMBER_OF_JOINED_BLOCKS = 10
# This option will show only the first 100 characters of each block of the joined document in the final output file

if RUN_ALL_EXTRACTORS:
    list_extractors = ['tfidf','kpminer','yake','textrank','singlerank','topicrank','positionrank','multipartiterank']
else:
    list_extractors = [EXTRACTOR_SELECTED]

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME = "output_KE_pke.xlsx"
COLUMN_PREFIX = 'keywords_pke'

# %%
# Brief description of the keyword extractors:
#
# Statistical models
#     TfIdf
#     KPMiner (El-Beltagy and Rafea, 2010) http://www.aclweb.org/anthology/S10-1041.pdf
#     YAKE (Campos et al., 2020) https://doi.org/10.1016/j.ins.2019.09.013
# Graph-based models
#     TextRank (Mihalcea and Tarau, 2004) http://www.aclweb.org/anthology/W04-3252.pdf
#     SingleRank (Wan and Xiao, 2008) http://www.aclweb.org/anthology/C08-1122.pdf
#     TopicRank (Bougouin et al., 2013) http://aclweb.org/anthology/I13-1062.pdf
#     PositionRank (Florescu and Caragea, 2017) http://www.aclweb.org/anthology/P17-1102.pdf
#     MultipartiteRank (Boudin, 2018) https://arxiv.org/abs/1803.08721


# More information: https://github.com/boudinfl/pke?tab=readme-ov-file#implemented-models

# %%
# Run this command in the terminal: python -m spacy download en_core_web_sm

# %%
import os
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import pke

folder_file_path = "./"

file_path = os.path.join(folder_file_path, FILENAME)
output_file_path = os.path.join(folder_file_path, OUTPUT_FILENAME)

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
if JOIN_DOCUMENTS_BEFORE_EXTRACTION:
    joined_text = ' '.join(df['texts'].tolist())

    def split_string(text, number_of_parts):
        total_length = len(text)
        
        # Calculate the size of each part (rounded up to ensure we cover the entire text)
        part_size = total_length // number_of_parts
        
        # Split the text into 10 parts
        parts = []
        for i in range(number_of_parts):
            start = i * part_size
            # For the last part, use the end of the text
            end = min((i + 1) * part_size, total_length)
            parts.append(text[start:end])
        
        return parts
    
    joined_df = pd.DataFrame(columns=['texts'])
    joined_df['texts'] = split_string(joined_text, NUMBER_OF_JOINED_BLOCKS)
    df = joined_df

# %%
df

# %%
def extract_keywords_pke(text):
    text = str(text)   # Convert to string to handle potential NaN values
    if pd.isna(text):
        return []     
    
    if EXTRACTOR_SELECTED=='tfidf':
        extractor = pke.unsupervised.TfIdf()
    elif EXTRACTOR_SELECTED=='kpminer':
        extractor = pke.unsupervised.KPMiner()
    elif EXTRACTOR_SELECTED=='yake':
        extractor = pke.unsupervised.YAKE()
    elif EXTRACTOR_SELECTED=='textrank':
        extractor = pke.unsupervised.TextRank()
    elif EXTRACTOR_SELECTED=='singlerank':
        extractor = pke.unsupervised.SingleRank()
    elif EXTRACTOR_SELECTED=='topicrank':
        extractor = pke.unsupervised.TopicRank()
    elif EXTRACTOR_SELECTED=='positionrank':
        extractor = pke.unsupervised.PositionRank()
    elif EXTRACTOR_SELECTED=='multipartiterank':
        extractor = pke.unsupervised.MultipartiteRank() 

    extractor.load_document(input=text, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()

    # Hide warnings
    import logging
    logger = logging.getLogger()
    logger.disabled = True

    keyphrases = extractor.get_n_best(n=NUMBER_OF_KEYWORDS)

    keyphrases_words = [keyphrase[0] for keyphrase in keyphrases]

    return sorted(keyphrases_words)

# Apply keyword extraction to the 'texts' column
for EXTRACTOR_SELECTED in list_extractors:
    df[COLUMN_PREFIX+'_'+EXTRACTOR_SELECTED] = df['texts'].progress_apply(extract_keywords_pke)

# %%
# Include only the first 100 characters of each block of the joined document
if JOIN_DOCUMENTS_BEFORE_EXTRACTION:
    df['texts'] = df['texts'].progress_apply(lambda x: x[:100])

# %%
df

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
    success = save_df_to_excel(df, output_file_path)
    if success:
        print(f"\nFile successfully saved and verified at: {output_file_path}")
    else:
        print(f"\nFile saved but verification failed. Please check the output manually.")
except Exception as e:
    print(f"Error saving the file: {e}")

# %%



