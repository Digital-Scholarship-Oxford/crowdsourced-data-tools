# %%
NER_categories = ['cardinal', 'date', 'event', 'facility', 'geopolitical', 'language', 
                  'law', 'location', 'monetary','affiliation', 'ordinal', 'organization', 
                  'percentage', 'people', 'product', 'measurement', 'artwork']

ML_MODEL = "gpt-4o-mini"
# Alternative models: 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1'

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME = "output_NER_OpenAI.xlsx"
COLUMN_PREFIX = 'entities_OpenAI'

OPENAI_API_KEY = "" # For security reasons, we recommend that you use the .env file to store the API key instead of using this variable.

# %%
if OPENAI_API_KEY == "":
    import os
    from dotenv import load_dotenv
    load_dotenv() # LOAD OPENAI API KEY from .env file

# %%
import os
import pandas as pd
import ast
from tqdm.auto import tqdm
tqdm.pandas()

import json
from openai import OpenAI


if OPENAI_API_KEY != "":
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = OpenAI()

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
def remove_empty_arrays(dictionary):
    return {k: v for k, v in dictionary.items() if not (isinstance(v, list) and len(v) == 0)}

def dict_to_sets(dictionary):
    return {key: set(value) for key, value in dictionary.items()}

properties_dict = {}
for category in NER_categories:
    properties_dict[category] = {
        "type": "array",
        "items": {"type": "string"}
    }

def extract_entities_openai(text):
    text = str(text)   # Convert to string to handle potential NaN values
    if pd.isna(text):
        return []
    
    response = client.responses.create(
        model=ML_MODEL,
        input=[
            {"role": "system", "content": "Extract entities from the input text"},
            {
                "role": "user",
                "content": text
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "entities",
                "schema": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": NER_categories,
                    "additionalProperties": False
                },
                "strict": True
            }
        },
    )

    response_output_text = json.loads(response.output_text)
    entities = dict_to_sets(remove_empty_arrays(response_output_text))
 
    # return [entities,response]
    return entities

# Apply entity extraction to the 'texts' column
openai_input = df['texts'].to_list()
openai_outputs = []
for text in tqdm(openai_input):
    try:
        openai_outputs.append(extract_entities_openai(text))
    except Exception as e:
        print(f"Error: {e}")
        openai_outputs.append('Error')

# %%
# Error indices
for idx,output in enumerate(openai_outputs):
    if isinstance(output, str) and output == 'Error':
        print(idx,openai_input[idx])

# %%
# # Retry error indices
# for idx,output in enumerate(openai_outputs):
#     if isinstance(output, str) and output == 'Error':
#         print(idx)
#         try:
#             openai_outputs[idx] = (extract_entities_openai(openai_input[idx]))
#         except Exception as e:
#             print(f"Error: {e}")

# # Error indices
# for idx,output in enumerate(openai_outputs):
#     if isinstance(output, str) and output == 'Error':
#         print(idx,openai_input[idx])

# %%
df[COLUMN_PREFIX+'_'+ML_MODEL] = openai_outputs

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



