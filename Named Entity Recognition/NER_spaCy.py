# %%
# ML_MODEL = "en_core_web_trf" 
ML_MODEL = "en_core_web_sm"
# Alternative models: "en_core_web_sm", "en_core_web_lg"

# To download the models, run the following commands:
# python -m spacy download en_core_web_trf
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_lg

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME = "output_NER_spaCy.xlsx"
COLUMN_PREFIX = 'entities_spaCy'

# %%
# Brief description of the models:

# NER 18 classes scheme: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, 
# LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, 
# QUANTITY, TIME, WORK_OF_ART

# More information in https://spacy.io/models/en

# %%
import os
import pandas as pd
import ast
from tqdm.auto import tqdm
tqdm.pandas()

import spacy

try:
    nlp = spacy.load(ML_MODEL)
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
def extract_entities_spacy(text):
    text = str(text)   # Convert to string to handle potential NaN values
    if pd.isna(text):
        return []    
    doc = nlp(text) 
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = set()
        entities[ent.label_].add(ent.text)
    return entities

# Apply entity extraction to the 'texts' column
df[COLUMN_PREFIX+'_'+ML_MODEL] = df['texts'].progress_apply(extract_entities_spacy)

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



