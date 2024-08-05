## Project
Generating Arguments by Instruction-Fine-Tuning LLMs on Factual Causal Questions

## Description
Ample research has been done on the factual accuracy of generated responses of LLMs and how the accuracy can be increased. However, there are open research questions regarding the factual accuracy of LLMs in generated arguments. Additionally, there is currently no available research on the knowledge retention capability of LLMs after instruction-fine-tuning and how the retained knowledge is recalled during argument generation. 

This work also includes a short experiment to probe the causal knowledge of LLMs before and after instruction-fine-tuning on causal question and answer pairs. The goal of this experiment is to find the strengths and weaknesses of LLMs to diverse questions. The experiment provides useful information on the process of capturing of knowledge.  

This repository consists of all the related papers that inspire this work, datasets constructed by extracting causal relations from CauseNet, questions-datasets constructed by adding questions to the extracted causal relationship, all the scripts associated with fine-tuning the model and preparing the dataset as well as extracting the causal relationship and constructing the questions-dataset. 

The repository is organized according to the sections in the thesis. Therefore, the folder 'Experiments' includes the approach and the scripts and the 'Result' folder includes the results of the evaluation. To save space, some large datasets have been compressed into ZIP-files.  

The fine-tuning is conducted on the 'alpaca-native' model, which is available in HuggingFace. Due to the large size of the data, the LLM has not been saved in the repo but can be accessed via [HuggingFace](https://huggingface.co/chavinlo/alpaca-native).  

## Experiments
The experiments section consists of 6 subsections: 
1. Section 5.1: Causal Relation extraction: The scripts and data pertaining to the extraction of causal relationships from CauseNet can be found in this subsection. Furthermore, the topics
2. Section 5.2: Topic selection: The topics from which the causal relationships will be extracted is listed in a CSV file in this folder.
3. Section 5.3: Question Dataset Construction: The Question Dataset constructed from causal relationships are stored in this section.
4. Section 5.4: Causal Knowledge Probing: Data relating to the causal knowledge probing can be found in this folder.
5. Section 5.5: Experimental Setup: Data relating to the fine-tuning set-up, datasets for fine-tuning and the scripts used for fine-tuning can be found here. 
6. Section 5.6: Evaluation: The scripts for computing automatic evaluation (BLEU, ROUGE, BERT-Score) are stored in this folder. The scripts to generate essays and arguments for manual evaluation are also stored in this folder.


## Section 5.1: Causal Relation extraction
The folder includes four Jupyter-Notebook scripts that extract the cause of the topic and the first, second and third-order effects of the topic. The resulting extracted causal relationships are stored in the 'Extracted Causal Relationships' folder. Additionally, the complete CauseNet dataset is provided.

## Section 5.2 Topic Selection
The folder includes the CSV file, which includes the topics extracted from the [The Argument Ontology](https://zenodo.org/records/5180409). 10 topics each from 14 different domains were chosen from TAO. ince not all domains had 10 topics, all topics were chosen from some domains. Therefore, the total number of topics is 127.

## Section 5.3: Question Dataset Construction
The folder is compressed due to the large data size. The question and answer templates are stored in this folder along with the scripts for generating the templates.

## Section 5.4: Causal Knowledge Probing
The folder includes the script for executing the Causal Knowledge Probing and the statistics resulting from the probing.

## Section 5.5: Experimental Setup
The dolfer includes the subfolder of Training Dataset comprising the training, validation and test dataset. The 'Scripts' subfolder includes the scripts for fine-tuning the model.  

## Section 5.6 Evaluation
The folder includes the scripts for the preparation of candidate and reference sentences for automatic evluation. The labels prepared for the automatic evaluation are also stored in the folder. The script for the automatic evaluation on the labels is also stored in this folder. Furthermore, the scripts to generate the essays for manual evaluation has been stored in this folder. 

## Results
The results include the adapters created after fine-tuning, results from the automatic and manual evaluation. The folder also includes the annotation data in the original form and in the numerical values form, which was required to compute the Inter-Annotator-Agreement.