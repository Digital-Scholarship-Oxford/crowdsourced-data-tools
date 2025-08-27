# %%
NER_categories = ['cardinal', 'date', 'event', 'facility', 'geopolitical', 'language', 
                  'law', 'location', 'monetary','affiliation', 'ordinal', 'organization', 
                  'percentage', 'people', 'product', 'measurement', 'artwork'] 

ML_MODEL = "knowledgator/modern-gliner-bi-large-v1.0"
# Alternative models: "urchade/gliner_large-v2.1", "gliner-community/gliner_large-v2.5", 
# "gliner-community/gliner_xxl-v2.5", "EmergentMethods/gliner_medium_news-v2.1"

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME = "output_NER_GLiNER.xlsx"
COLUMN_PREFIX = 'entities_GLiNER'


# %%
# Brief description of the models:

# open-type NER BERT/GLiNER Model by the authors of https://arxiv.org/pdf/2406.10258 :
# "EmergentMethods/gliner_medium_news-v2.1" 

# open-type NER BERT/GLiNER Model by authors of https://arxiv.org/abs/2311.08526 :
# "urchade/gliner_large-v2.1" 
# "knowledgator/modern-gliner-bi-large-v1.0" 
# "gliner-community/gliner_large-v2.5" 
# "gliner-community/gliner_xxl-v2.5" 

# %%
import os
import pandas as pd
import ast
from tqdm.auto import tqdm
tqdm.pandas()

import torch
from gliner import GLiNER


model = GLiNER.from_pretrained(ML_MODEL)

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
def extract_entities_gliner(text):
    text = str(text)   # Convert to string to handle potential NaN values
    if pd.isna(text):
        return []
    
    text = [x for x in text.split('\n') if x != ""]
    
    entities = {}
    for sentence in text:
        entities_pred = model.predict_entities(sentence, NER_categories)
        for entity in entities_pred:          
            if entity['label'] not in entities:
                 entities[entity['label']] = set()
            entities[entity['label']].add(entity['text'])
    return entities

# Apply entity extraction to the 'texts' column
df[COLUMN_PREFIX+'_'+ML_MODEL] = df['texts'].progress_apply(extract_entities_gliner)

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



