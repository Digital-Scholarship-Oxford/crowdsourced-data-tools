# %%
NUMBER_OF_KEYWORDS = 5

EXTRACTOR_SELECTED = "gpt-4o-mini"
# Alternative models: 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1'

JOIN_DOCUMENTS_BEFORE_EXTRACTION = False  # Do not change without checking the context length of the extractor !
NUMBER_OF_JOINED_BLOCKS = 10
# This option will show only the first 100 characters of each block of the joined document in the final output file

FILENAME = "data_sample_input.xlsx"
OUTPUT_FILENAME = "output_KE_OpenAI.xlsx"
COLUMN_PREFIX = 'keywords_OpenAI'


OPENAI_API_KEY = "" # For security reasons, we recommend that you use the .env file to store the API key instead of using this variable.

# %%
if OPENAI_API_KEY == "":
    import os
    from dotenv import load_dotenv
    load_dotenv() # LOAD OPENAI API KEY from .env file

# %%
import os
import pandas as pd
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
def remove_empty_arrays(dictionary):
    return {k: v for k, v in dictionary.items() if not (isinstance(v, list) and len(v) == 0)}

def dict_to_sets(dictionary):
    return {key: set(value) for key, value in dictionary.items()}

properties_dict = {}

properties_dict['keywords'] = {
    "type": "array",
    "items": {"type": "string"}
}

def extract_keywords_openai(text):
    text = str(text)   # Convert to string to handle potential NaN values
    if pd.isna(text):
        return []
    
    response = client.responses.create(
        model=EXTRACTOR_SELECTED,
        input=[
            {"role": "system", "content": "Extract "+str(NUMBER_OF_KEYWORDS)+" representative keywords from the input text"},
            {
                "role": "user",
                "content": text
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "keywords",
                "schema": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": ["keywords"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
    )

    response_output_text = json.loads(response.output_text)
    keywords = dict_to_sets(remove_empty_arrays(response_output_text))
 
    # return [keywords,response]
    return keywords

# Apply entity extraction to the 'texts' column
openai_input = df['texts'].to_list()
openai_outputs = []
for text in tqdm(openai_input):
    try:
        openai_outputs.append(extract_keywords_openai(text))
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
#             openai_outputs[idx] = (extract_keywords_openai(openai_input[idx]))
#         except Exception as e:
#             print(f"Error: {e}")

# # Error indices
# for idx,output in enumerate(openai_outputs):
#     if isinstance(output, str) and output == 'Error':
#         print(idx,openai_input[idx])

# %%
# Transform the output to a list of keywords
openai_outputs =[list(el['keywords']) for el in openai_outputs]

# %%
df[COLUMN_PREFIX+'_'+EXTRACTOR_SELECTED] = openai_outputs

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



