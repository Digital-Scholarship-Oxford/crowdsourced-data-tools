# Extracting Keywords from Crowdsourced Collections Project 

## Content Description 

AI tools developed at the University of Oxford for analysing crowdsourced text collections. They enable the identification of keywords, topics, and categories of terms. 

Project details can be found under 'Project Summary' and 'Methodology'. 

### Who is it for? 
While this repository was developed with colleagues working in the Digital Humanities (DH) and the Galleries, Libraries, Archives and Museums (GLAM) sector in mind, we welcome and encourage re-use of its content by anyone.  

### What's here? 
This repository contains **x3 folders** which support the use of the following AI-powered text analysis approaches: 

 #### • [Keyword Extraction (KE)](https://github.com/Digital-Scholarship-Oxford/crowdsourced-data-tools/tree/main/Keyword%20Extraction)
[Documentation](https://github.com/Digital-Scholarship-Oxford/crowdsourced-data-tools/blob/main/Keyword%20Extraction/README_KE.md)

KE is used to extract keywords directly from text, words or phrases, which both e.g. arise in a document and indicate what it is that the text is talking about ([Nomoto, 2023](https://link.springer.com/article/10.1007/s42979-022-01481-7)). 
 
 #### • [Named Entity Recognition (NER)](https://github.com/Digital-Scholarship-Oxford/crowdsourced-data-tools/tree/main/Named%20Entity%20Recognition) 
[Documentation](https://github.com/Digital-Scholarship-Oxford/crowdsourced-data-tools/blob/main/Named%20Entity%20Recognition/README_NER.md)

NER is used to identify segments of information referenced in a text and classify them into pre-established categories, such as 'person', 'organisation' and 'location' ([Jehangir et al, 2023](https://www.sciencedirect.com/science/article/pii/S2949719123000146#b11)) - though the range of categories predicted can vary from model to model.  
 
 *Example:* <br>
Text = 'An author called Jane Austen wrote the novel Pride and Prejudice.'   
NER model output = Person: Jane Austen; Work of Art: Pride and Prejudice 

 #### • [Topic Modelling (TM)](https://github.com/Digital-Scholarship-Oxford/crowdsourced-data-tools/tree/main/Topic%20Modelling)
[Documentation](https://github.com/Digital-Scholarship-Oxford/crowdsourced-data-tools/blob/main/Topic%20Modelling/README_Topic.md)

TM is used to uncover hidden thematic structures in large collections of text/textual documents, providing an automatic means to organise, understand and summarise them. It is a type of unsupervised machine learning that supports analysis of unstructured textual data i.e. it does not rely on labelled input.
<br><br>

**In each folder you will find:**

• An approach-specific README (`.md`) 

• Requirements information (`.txt`) e.g. python libraries needed 

• A data sample (`.xlsx`) 

• `.py` & `.ipynb` script files  

These files support the use of different models (both open and commercial), and can be used either locally or via e.g. [Google Colab](https://colab.research.google.com/) - the latter being recommended for those using institutionally managed devices or who have less experience working with code.  
<br><br>
## Project Summary 

**Extracting Keywords from Crowdsourced Collections** was a Digital Scholarship @ Oxford (DiSc) Research Development Grant-funded project based in the Faculty of English at the University of Oxford, which ran from **November 2024 to July 2025**.  

Using the **Their Finest Hour Online Archive**, a digital collection of 2,000+ records and 26,000+ files related to the Second World War, as a case study, this project set out to explore how Natural Language Processing (NLP) methods could be utilised to extract keywords from crowdsourced digital collection data.  

Assigning appropriate keyword tags to digital collection records is a crucial step in supporting search and discovery, as well as adherence to FAIR data principles. Traditionally, this process has involved manually assigning keywords, often using a pre-defined/inherited controlled vocabulary used within a particular institution. Manual tagging of keywords can be resource-intensive, potentially lead to the misrepresentation of records or collections, and can perpetuate historic assumptions, biases and stereotypes associated with particular domains. While there have been efforts to democratise digital collections metadata creation, in the case of keyword tagging, the primary assumption that underpins this process remains the same: individuals should select and then impose keywords on historical data. 

This project sought to invert that assumption, and explore the extent to which NLP methods and tools can be used to allow collection records to generate their own keyword tags, and thus describe or 'speak for' themselves. This is particularly relevant to crowdsourced collections of personal histories, especially Second World War collections, at a time when representations of the past are being reshaped to serve political interests. <br><br>

## People 

**Stuart Lee**, Principal Investigator. stuart.lee@it.ox.ac.uk  

**Matthew Kidd**, Research Assistant. matthew.kidd@conted.ox.ac.uk  

**Catherine Conisbee**, Research Assistant. catherine.conisbee@bodleian.ox.ac.uk  

**Miguel Arana-Catania**, Senior Research Software Engineer AI/NLP. miguel.arana-catania@humanities.ox.ac.uk  <br><br>

## Project Goals: 

1) To explore/develop an NLP workflow for keyword extraction from crowdsourced digital collections data 

2) To understand how programmatic extraction of keywords can enhance FAIR principles in crowdsourced collections to maximise discoverability and research value 

3) To explore whether and how enhanced keyword tagging processes can advance research possibilities and understanding of archive collections e.g. experiences of WW2  

4) To explore to what extent enhanced keyword extraction has the potential to globalise WW2 collections 

5) To build a collaborative network  

6) To report back and share findings <br><br>

## Project Methodology 

During the course of our research, the project team identified four core approaches to explore.  
In each case the models were tested exclusively against the 'description' fields of Their Finest Hour (TFH) Online Archive records. 

While predominantly open models were utilised for testing, the team also experimented with commercial models such as OpenAI's gpt-4o and gpt-4o-mini for each approach for fullness. 
 
#### Named Entity Recognition (NER)  
We tested 20 NER models using a small sample of records from the TFH archive, evaluating them against manually curated entity lists, assessing both the accuracy of models in identifying and categorising entities. 
 

#### Keyword Extraction (KE) 
We tested 14 KE models using a small sample of records from the TFH archive and evaluated them against a manually curated keyword list, assessing the accuracy of models in identifying keywords. 

In this case, we also briefly experimented with llama-2-7b-chat. Q4_K_M, prompted to extract keywords, from all record descriptions. 
 

#### Topic Modelling (TM) 
We tested 9 models against all record descriptions contained in the Their Finest Hour Online Archive.  
 
We also experimented with the use of generative AI models – Gemini 2.5, GPT-4o, Claude Sonnet 4 – to assign human-readable labels to the word clusters (i.e. topics) identified by the models. 

#### Emotion Classification (EC) 
We tested 6 emotion classification models, and Gemini 2.5, using the full dataset, with outputs evaluated in detail for 25 stories. <br><br>

## License & Citation 

The content of this repository is being made available under an [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html) re-use license. <br><br>

## Resources and Links 

**Their Finest Hour Online Archive:** https://portal.sds.ox.ac.uk/Their_Finest_Hour 

**Their Finest Hour Project Website:** https://theirfinesthour.english.ox.ac.uk/  

**Digital Scholarship @ Oxford:** https://digitalscholarship.web.ox.ac.uk/
