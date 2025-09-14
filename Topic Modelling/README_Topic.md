The code in this folder allows finding the topics of a set of texts and organising the texts according to such topics (a process usually called Topic Modelling).

# Quick Start

For a quick start, install all the required libraries:

```shell
pip install -r requirements.txt
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
```

In this case, you can ignore the rest of this section.

To install the required libraries for a particular version, run the following commands, depending on which code you want to use:

- sklearn: `pip install openpyxl pandas tqdm spacy scikit-learn numpy` 

- BERTopic: `pip install openpyxl pandas tqdm spacy scikit-learn numpy bertopic` 

- OpenAI: `pip install openpyxl pandas tqdm spacy scikit-learn numpy bertopic python-dotenv openai pillow` The OpenAI models require you to have an OpenAI key, which should be specified either in the variable OPENAI_API_KEY or in the "env" file, which should be renamed to ".env"


If you experience any problems, you can specify the tested version of the libraries (using Python=3.11):

- sklearn: `pip install openpyxl==3.1.5 pandas==2.3.1 tqdm==4.67.1 spacy==3.8.7 scikit-learn==1.7.1 numpy==2.3.1` 

- BERTopic: `pip install openpyxl==3.1.5 pandas==2.3.1 tqdm==4.67.1 spacy==3.8.7 scikit-learn==1.7.1 numpy==2.2.6 bertopic==0.17.3` 

- OpenAI: `pip install openpyxl==3.1.5 pandas==2.3.1 tqdm==4.67.1 spacy==3.8.7 scikit-learn==1.7.1 numpy==2.2.6 bertopic==0.17.3 python-dotenv==1.1.1 openai==1.99.1 pillow==11.3.0` The OpenAI models require you to have an OpenAI key, which should be specified either in the variable OPENAI_API_KEY or in the "env" file, which should be renamed to ".env"

For more information about the installation of the libraries, consult the following links:

- sklearn: [sklearn documentation](https://scikit-learn.org/stable/user_guide.html) and [sklearn GitHub](https://github.com/scikit-learn/scikit-learn)

- BERTopic: [BERTopic documentation](https://maartengr.github.io/BERTopic/index.html) and [BERTopic GitHub](https://github.com/MaartenGr/BERTopic)

- OpenAI: [OpenAI developer documentation](https://platform.openai.com/docs/overview)


## Using the code

### Input data

These scripts require the text data to be provided in an Excel file. The file should contain just a single column with the text. The text can be split in multiple rows.

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

`NUMBER_OF_TOPICS = 5`  Number of topics in which to categorise all the texts.

`NUMBER_OF_TERMS_PER_TOPIC = 8`  Number of words to define each topic.

`NUMBER_OF_DOCS_PER_TOPIC = 3`  Number of most representative documents to show for each topic.

`FILENAME = "data_sample_input.xlsx"`  Name of the Excel file with the input data.

`OUTPUT_FILENAME_TOPICS = "output_Topic_topics_BERTopic.xlsx"` Name of the Excel file which will be created with the topics and their description, after running the code.

`OUTPUT_FILENAME_DOCS_DISTRIB = "output_Topic_topics_BERTopic.xlsx"` Name of the Excel file which will be created with the most relevant documents for each topic, after running the code.

`INSTALL_REQUIRED_LIBRARIES = False`  Change to True if you want the scripts to automatically install the required libraries.

`USE_GOOGLE_DRIVE = False`  Change to True if you want the code to be run in Google Colab. You should also change the previous variable to True to install the libraries automatically.

`ML_MODEL = 'all-MiniLM-L6-v2'`  Machine learning to be used.


## Machine learning models



More information: https://github.com/boudinfl/pke?tab=readme-ov-file#implemented-models

- sklearn: 


'svd' https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis

'nmf' https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf

'lda' https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html


- BERTopic: 

https://maartengr.github.io/BERTopic/index.html

https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html#sentence-transformers


- OpenAI: 

Information about the models: https://platform.openai.com/docs/overview


# Citing this code

TBD

# Contact

Miguel Arana-Catania. Senior Research Software Engineer AI/NLP, University of Oxford. miguel.arana-catania@humanities.ox.ac.uk

Matthew Kidd.

Catherine Conisbee. Research Assistant, Extracting Keywords from Crowdsourced Collections Project, University of Oxford. catherine.conisbee@bodleian.ox.ac.uk

# License

TBD





