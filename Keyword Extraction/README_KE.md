The code in this folder allows the extraction of keywords from a set of texts. This process is usually called Keyword Extraction (KE).

# Quick Start

For a quick start, install all the required libraries:

(Git needs to be installed before proceeding. See https://git-scm.com/downloads or if using Anaconda do `conda install git`)

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The input data should be in an Excel file in the same folder, by default named `data_sample_input.xlsx`. A single column with the text, which can be split into multiple rows.

Run any of the Python scripts:

```shell
python <script_name>.py
```

or the equivalent interactive Python notebooks using your favourite IDE/platform (Jupyter Notebook, VS CODE, Google Colab, etc.). You may need to install the ipykernel library (`pip install ipykernel`)

Edit the Python scripts or interactive notebooks to change the default options defined at the top.

More detail in the following sections.

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

Install Git before proceeding. See https://git-scm.com/downloads or if using Anaconda do `conda install git`

If you want to install all the required libraries for all the versions of the code, use:

```shell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

In this case, you can ignore the rest of this section.

To install the required libraries for a particular version, run the following commands, depending on which code you want to use:

- pke: `pip install pandas tqdm openpyxl numpy` and `pip install git+https://github.com/boudinfl/pke.git` and `python -m spacy download en_core_web_sm`

- RaKUn: `pip install pandas tqdm openpyxl numpy rakun2` 

- YAKE: `pip install pandas tqdm openpyxl numpy` and `pip install git+https://github.com/LIAAD/yake`

- KeyBERT: `pip install pandas tqdm openpyxl numpy keybert`

- OpenAI: `pip install python-dotenv openai openpyxl pandas tqdm pillow` The OpenAI models require you to have an OpenAI key, which should be specified either in the variable OPENAI_API_KEY or in the "env" file, which should be renamed to ".env"


If you experience any problems, you can specify the tested version of the libraries (using Python=3.11):

- pke: `pip install pandas==2.3.1 tqdm==4.67.1 openpyxl==3.1.5 numpy==2.3.1` and `pip install git+https://github.com/boudinfl/pke.git`  and `python -m spacy download en_core_web_sm`

- RaKUn: `pip install pandas==2.3.1 tqdm==4.67.1 openpyxl==3.1.5 numpy==2.3.1 rakun2==0.31` 

- YAKE: `pip install pandas==2.3.1 tqdm==4.67.1 openpyxl==3.1.5 numpy==2.3.1` and `pip install git+https://github.com/LIAAD/yake`

- KeyBERT: `pip install pandas==2.3.1 tqdm==4.67.1 openpyxl==3.1.5 numpy==2.3.1 keybert==0.9.0`

- OpenAI: `pip install python-dotenv==1.1.1 openai==1.98.0 openpyxl==3.1.5 pandas==2.3.1 tqdm==4.67.1 pillow==11.3.0` The OpenAI models require you to have an OpenAI key, which should be specified either in the variable OPENAI_API_KEY or in the "env" file, which should be renamed to ".env"

For more information about the installation of the libraries, consult the following links:

- pke: [pke GitHub](https://github.com/boudinfl/pke)

- RaKUn: [RaKUn GitHub](https://github.com/skblaz/rakun2)

- YAKE: [YAKE GitHub](https://github.com/LIAAD/yake) and [YAKE documentation](https://liaad.github.io/yake/docs/--home)

- KeyBERT: [KeyBERT GitHub](https://github.com/MaartenGr/KeyBERT) and [KeyBERT documentation](https://maartengr.github.io/KeyBERT/)

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

`NUMBER_OF_KEYWORDS = 5`  Number of keywords to be extracted from each text.

`FILENAME = "data_sample_input.xlsx"`  Name of the Excel file with the input data.

`OUTPUT_FILENAME = "output_KE_pke.xlsx"` Name of the Excel file which will be created with the results, after running the code.

`INSTALL_REQUIRED_LIBRARIES = False`  Change to True if you want the scripts to automatically install the required libraries.

`USE_GOOGLE_DRIVE = False`  Change to True if you want the code to be run in Google Colab. You should also change the previous variable to True to install the libraries automatically.

`JOIN_DOCUMENTS_BEFORE_EXTRACTION = False`  Change to True if you want all the text rows to be joined to extract the keywords together. By default, the joined text will be split into 10 blocks of text.

`EXTRACTOR_SELECTED = 'paraphrase-mpnet-base-v2'`  Keyword extractor to be used.


## Keyword extraction methods

- pke: 

-Statistical models:

TfIdf

KPMiner (El-Beltagy and Rafea, 2010) http://www.aclweb.org/anthology/S10-1041.pdf

YAKE (Campos et al., 2020) https://doi.org/10.1016/j.ins.2019.09.013

-Graph-based models:

TextRank (Mihalcea and Tarau, 2004) http://www.aclweb.org/anthology/W04-3252.pdf

SingleRank (Wan and Xiao, 2008) http://www.aclweb.org/anthology/C08-1122.pdf

TopicRank (Bougouin et al., 2013) http://aclweb.org/anthology/I13-1062.pdf

PositionRank (Florescu and Caragea, 2017) http://www.aclweb.org/anthology/P17-1102.pdf

MultipartiteRank (Boudin, 2018) https://arxiv.org/abs/1803.08721


More information: https://github.com/boudinfl/pke?tab=readme-ov-file#implemented-models

- RaKUn: 

https://github.com/skblaz/rakun2 

https://link.springer.com/chapter/10.1007/978-3-031-18840-4_27

- YAKE: 

https://liaad.github.io/yake/docs/--home  

https://github.com/LIAAD/yake

- KeyBERT: 

https://github.com/MaartenGr/KeyBERT 

https://maartengr.github.io/KeyBERT/


- OpenAI: 

Information about the models: https://platform.openai.com/docs/overview


# Citing this code

TBD

# Contact

Miguel Arana-Catania. Senior Research Software Engineer AI/NLP, University of Oxford. miguel.arana-catania@humanities.ox.ac.uk

Matthew Kidd.

Catherine Conisbee.

# License

TBD





