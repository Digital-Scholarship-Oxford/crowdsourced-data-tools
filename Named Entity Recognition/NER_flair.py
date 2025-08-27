# %%
ML_MODEL = 'ner-large'
# Alternative models: 'ner-fast', 'ner-ontonotes-fast', 'ner-ontonotes-large'

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME = "output_NER_flair.xlsx"
COLUMN_PREFIX = 'entities_flair'

# %%
# Brief description of the models:

# 'ner-large' # 4 entity classes. Model based on document-level XLM-RoBERTa embeddings and FLERT.
# 'ner-fast' # 4 entity classes. Model based on flair embeddings and LSTM-CRF.
# 'ner-ontonotes-fast' # 18 entity classes. Model based on flair embeddings and LSTM-CRF.
# 'ner-ontonotes-large' # 18 entity classes. Model based on document-level XLM-RoBERTa embeddings and FLERT.

# NER 4 classes scheme: PER, LOC, ORG,MISC

# NER 18 classes scheme: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, 
# LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, 
# QUANTITY, TIME, WORK_OF_ART

# More information: https://flairnlp.github.io/docs/tutorial-basics/tagging-entities#list-of-ner-models



# %%
import os
import pandas as pd
import ast
from tqdm.auto import tqdm
tqdm.pandas()

from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

try:
    # load the model
    tagger = Classifier.load(ML_MODEL)
    splitter = SegtokSentenceSplitter()
except OSError:
    print("Model not downloaded")

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
df

# %%
def extract_entities_flair(text):    
    text = str(text)   # Convert to string to handle potential NaN values
    if pd.isna(text):
        return []
        
    text = text.strip()
    sentences = splitter.split(text)

    tagger.predict(sentences) 
    entities = {}
    for sentence in sentences:
        for ent in sentence.to_dict()['entities']:
            for label in ent['labels']:
                if label['value'] not in entities:
                    entities[label['value']] = set()
                entities[label['value']].add(ent['text'])
    return entities


# Apply entity extraction to the 'texts' column
df[COLUMN_PREFIX+'_'+ML_MODEL] = df['texts'].progress_apply(extract_entities_flair)

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



