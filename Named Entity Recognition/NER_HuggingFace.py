# %%
ML_MODEL = "Babelscape/cner-base"
# Alternative models: 
# "xlm-roberta-large-finetuned-conll03-english",
# "dbmdz/bert-large-cased-finetuned-conll03-english",
# "Babelscape/wikineural-multilingual-ner",
# "elastic/distilbert-base-uncased-finetuned-conll03-english",
# "elastic/distilbert-base-cased-finetuned-conll03-english",
# "tner/roberta-large-wnut2017",
# "tner/deberta-large-wnut2017"

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME = "output_NER_HuggingFace.xlsx"
COLUMN_PREFIX = 'entities_HuggingFace'


# %%
# Brief description of the models:

# NER 4 classes scheme: PER, LOC, ORG, MISC for the following models:
# "xlm-roberta-large-finetuned-conll03-english" # RoBERTa Model by FacebookAI
# "dbmdz/bert-large-cased-finetuned-conll03-english" # BERT Model by the Bayerische Staatsbibliothek https://github.com/dbmdz 
# "Babelscape/wikineural-multilingual-ner" # mBERT Model by company Babelscape
# "elastic/distilbert-base-uncased-finetuned-conll03-english" # DistilBERT Model by company Elastic
# "elastic/distilbert-base-cased-finetuned-conll03-english" # DistilBERT Model by company Elastic

# NER 6 classes scheme: 'corporation', 'creative-work', 'group', 'location', 'person', 'product' for the following models:
# "tner/roberta-large-wnut2017" # RoBERTa Model by T-NER library https://github.com/asahi417/tner
# "tner/deberta-large-wnut2017" # DeBERTa Model by T-NER library https://github.com/asahi417/tner

# NER 29 classes scheme: 'ANIMAL', 'ARTIFACT', 'ASSET', 'BIOLOGY', 'CELESTIAL', 'CULTURE', 'DATETIME', 
# 'DISCIPLINE', 'DISEASE', 'EVENT', 'FEELING', 'FOOD', 'GROUP', 'LANGUAGE', 'LAW', 'LOC', 'MEASURE', 
# 'MEDIA', 'MONEY', 'ORG', 'PART', 'PER', 'PLANT', 'PROPERTY', 'PSYCH', 'RELATION', 'STRUCT', 
# 'SUBSTANCE', 'SUPER' for the following model:
# "Babelscape/cner-base" # DeBERTa Model by company Babelscape

# %%
import os
import pandas as pd
import ast
from tqdm.auto import tqdm
tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained(ML_MODEL)
model = AutoModelForTokenClassification.from_pretrained(ML_MODEL)
ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True,
               aggregation_strategy="simple")  

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
def extract_entities_hf(text):
    text = str(text)   # Convert to string to handle potential NaN values
    if pd.isna(text):
        return []
    
    text = [x for x in text.split('\n') if x != ""]
    ner_text = ner(text)
    
    entities = {}
    for sentence in ner_text:
        for entity in sentence:          
            if entity['entity_group'] not in entities:
                 entities[entity['entity_group']] = set()
            entities[entity['entity_group']].add(entity['word'])
    return entities

# Apply entity extraction to the 'texts' column
df[COLUMN_PREFIX+'_'+ML_MODEL] = df['texts'].progress_apply(extract_entities_hf)

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



