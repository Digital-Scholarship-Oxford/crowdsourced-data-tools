The code in this folder allows the extraction of entities from a set of texts. This process is usually called Named Entity Recognition (NER).

# Quick Start

For a quick start, install all the required libraries:

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

The input data should be in an Excel file in the same folder, by default named `data_sample_input.xlsx`. A single column with the text, which can be split into multiple rows.

Run any of the Python scripts:

```shell
python <script_name>.py
```

or the equivalent interactive Python notebooks using your favourite IDE/platform (Jupyter Notebook, VS CODE, Google Colab, etc.). You may need to install the ipykernel library (`pip install ipykernel`)

Edit the Python scripts or interactive notebooks to change the default options defined at the top.

More details in the following sections.

## Installation

### Virtual environment

It is not required, but you may install the required libraries in a virtual environment (e.g. [Anaconda](https://www.anaconda.com/)).

- To create and activate a virtual environment in Anaconda:

```shell
conda create --name <virtual_env_name> python=3.11
conda activate <virtual_env_name>
```

- To create and activate a virtual environment using the Python virtual environment module:

```shell
python3 -m venv "<virtual_env_name>"
source <virtual_env_name>/bin/activate
```

### Installing required libraries

If you want to install all the required libraries for all the versions of the code, use:

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

In this case, you can ignore the rest of this section.

To install the required libraries for a particular version, run the following commands, depending on which code you want to use:

- flair: `pip install flair pandas tqdm openpyxl numpy`

- GLiNER: `pip install transformers torch openpyxl gliner pandas tqdm` 

- HuggingFace: `pip install transformers torch openpyxl pandas tqdm`

- SpaCy: `pip install spacy openpyxl pandas tqdm`

- OpenAI: `pip install python-dotenv openai openpyxl pandas tqdm pillow` The OpenAI models require you to have an OpenAI key, which should be specified either in the variable OPENAI_API_KEY or in the "env" file, which should be renamed to ".env"


If you experience any problems, you can specify the tested version of the libraries (using Python=3.11):

- flair: `pip install flair==0.15.1 pandas==2.3.1 tqdm==4.67.1 openpyxl==3.1.5 numpy==2.3.1`

- GLiNER: `pip install transformers==4.54.1 torch==2.7.1 openpyxl==3.1.5 gliner==0.2.21 pandas==2.3.1 tqdm==4.67.1`

- HuggingFace: `pip install transformers==4.54.1 torch==2.7.1 openpyxl==3.1.5 pandas==2.3.1 tqdm==4.67.1`

- SpaCy: `!pip install spacy==3.8.7 openpyxl==3.1.5 pandas==2.3.1 tqdm==4.67.1` and `python -m spacy download en_core_web_trf` 

- OpenAI: `pip install python-dotenv==1.1.1 openai==1.98.0 openpyxl==3.1.5 pandas==2.3.1 tqdm==4.67.1 pillow==11.3.0` The OpenAI models require you to have an OpenAI key, which should be specified either in the variable OPENAI_API_KEY or in the "env" file, which should be renamed to ".env"

For more information about the installation of the libraries, consult the following links:

- flair: [flair GitHub](https://github.com/flairNLP/flair) and [flair documentation](https://flairnlp.github.io/flair/)

- GLiNER: [HuggingFace documentation](https://huggingface.co/docs) and [HuggingFace GLiNER documentation](https://huggingface.co/knowledgator/modern-gliner-bi-large-v1.0)

- HuggingFace: [HuggingFace documentation](https://huggingface.co/docs)

- SpaCy: [SpaCy GitHub](https://github.com/explosion/spaCy) and [SpaCy documentation](https://spacy.io/usage)

- OpenAI: [OpenAI developer documentation](https://platform.openai.com/docs/overview)


## Using the code

### Input data

These scripts require the text data to be provided in an Excel file. The file should contain just a single column with the text. The text can be split into multiple rows.

The default name of the file is `data_sample_input.xlsx`. This, as well as other options, can be customised. See the [customisation](#customisation) section.

If you are using Google Colab, by default, the input file should be in the main directory of your drive.

A sample input file is provided.

### Running the code

Each script comes in two versions: a Python script (.py) and an interactive Python notebook (.ipynb). Both versions are equivalent; use the one that you find more convenient.

To run the Python scripts, do:
```shell
python <script_name>.py
```

To run the interactive Python notebooks, use your favourite IDE/platform (Jupyter Notebook, VS CODE, Google Colab, etc.). You may need to install the ipykernel library (`pip install ipykernel`)


### Customisation

If you edit the scripts, you will find at the top of the script several variables that define options which can be customised. Examples of such variables are the following:

`FILENAME = "data_sample_input.xlsx"`  Name of the Excel file with the input data.

`OUTPUT_FILENAME = "output_NER_flair.xlsx"` Name of the Excel file which will be created with the results, after running the code.

`INSTALL_REQUIRED_LIBRARIES = False`  Change to True if you want the scripts to automatically install the required libraries.

`USE_GOOGLE_DRIVE = False`  Change to True if you want the code to be run in Google Colab. You should also change the previous variable to True to install the libraries automatically.

`NER_categories = ['cardinal', 'date', 'event', ...]`  If you use any open-entities model that allows defining the entities, here you can define them. 

`ML_MODEL = 'ner-large'`  Machine learning model to be used.


## Machine learning models

- flair: 

'ner-large' # 4 entity classes. Model based on document-level XLM-RoBERTa embeddings and FLERT.

'ner-fast' # 4 entity classes. Model based on flair embeddings and LSTM-CRF.

'ner-ontonotes-fast' # 18 entity classes. Model based on flair embeddings and LSTM-CRF.

'ner-ontonotes-large' # 18 entity classes. Model based on document-level XLM-RoBERTa embeddings and FLERT.

NER 4 classes scheme: PER, LOC, ORG, MISC

NER 18 classes scheme: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, 
LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, 
QUANTITY, TIME, WORK_OF_ART

More information: https://flairnlp.github.io/docs/tutorial-basics/tagging-entities#list-of-ner-models

- GLiNER: 


open-type NER BERT/GLiNER Model by the authors of https://arxiv.org/pdf/2406.10258 :

"EmergentMethods/gliner_medium_news-v2.1" 

open-type NER BERT/GLiNER Model by authors of https://arxiv.org/abs/2311.08526 :

"urchade/gliner_large-v2.1" 

"knowledgator/modern-gliner-bi-large-v1.0" 

"gliner-community/gliner_large-v2.5" 

"gliner-community/gliner_xxl-v2.5" 

- HuggingFace: 

NER 4 classes scheme: PER, LOC, ORG, MISC for the following models:

"xlm-roberta-large-finetuned-conll03-english" # RoBERTa Model by FacebookAI

"dbmdz/bert-large-cased-finetuned-conll03-english" # BERT Model by the Bayerische Staatsbibliothek https://github.com/dbmdz 

"Babelscape/wikineural-multilingual-ner" # mBERT Model by company Babelscape

"elastic/distilbert-base-uncased-finetuned-conll03-english" # DistilBERT Model by company Elastic

"elastic/distilbert-base-cased-finetuned-conll03-english" # DistilBERT Model by company Elastic

NER 6 classes scheme: 'corporation', 'creative-work', 'group', 'location', 'person', 'product' for the following models:

"tner/roberta-large-wnut2017" # RoBERTa Model by T-NER library https://github.com/asahi417/tner

"tner/deberta-large-wnut2017" # DeBERTa Model by T-NER library https://github.com/asahi417/tner


NER 29 classes scheme: 'ANIMAL', 'ARTIFACT', 'ASSET', 'BIOLOGY', 'CELESTIAL', 'CULTURE', 'DATETIME', 
'DISCIPLINE', 'DISEASE', 'EVENT', 'FEELING', 'FOOD', 'GROUP', 'LANGUAGE', 'LAW', 'LOC', 'MEASURE', 
'MEDIA', 'MONEY', 'ORG', 'PART', 'PER', 'PLANT', 'PROPERTY', 'PSYCH', 'RELATION', 'STRUCT', 
'SUBSTANCE', 'SUPER' for the following model:

"Babelscape/cner-base" # DeBERTa Model by company Babelscape

- SpaCy: 

NER 18 classes scheme: CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, 
LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, 
QUANTITY, TIME, WORK_OF_ART

More information in https://spacy.io/models/en

- OpenAI: 

Information about the models: https://platform.openai.com/docs/overview


# Citing this code

TBD

# Contact

TBD

# License

TBD





