from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import pandas as pd
import csv
import nltk


def generate_prompt(instruction: str, input_ctxt: str = None) -> str: #standard prompting made available in HuggingFace
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with a quesion. Write a response that appropriately answers the question.
        ### Instruction:
        {instruction}
        ### Input:
        {input_ctxt}
      	### Response:"""
    else:
      return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    	 ### Instruction:
      {instruction}
       ### Response:"""


model = "/bigwork/nhwpshaa/Fine-tuned AlpacaNative/Jobs/5 epochs/checkpoint-960" #load the model
tokenizer = LlamaTokenizer.from_pretrained("/bigwork/nhwpshaa/alpaca-native") #load the tokenizer
pipeline = transformers.pipeline( #initialize piepline
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


def loadDataset():
    test= pd.read_csv("/Experiments/Section 5.5: Experimental Setup/test.csv", sep=",", encoding="utf-8") #load the test dataset
    dataset = {"test": test }
    return dataset


def generateTrainingInstructions(dataset): #generate instructions specifically for training
    data = {"instruction":[], "input":[], "output":[]}
    for i, record in dataset.iterrows():
        question = record['Question']
        answer = record['Answer']
        if record['Question Type'].lower() == "yes/no question":
            data["instruction"].append(f"Answer the following questions with 'Yes' or 'No'")
        else:
            data["instruction"].append(f"Answer the following questions to the best of your knowledge. Answer the questions as briefly as possible using only one causal sentence.")
        data["output"].append(answer)
        data["input"].append(question)
    return data


dataset = loadDataset()
dataset = generateTrainingInstructions(dataset['test'])
labels= []
preds = []
instruction = "Act as an expert in the domain from which the following question stems from and answer the question in a single causal sentence"
for instruction, question, answer in zip(dataset['instruction'], dataset['input'], dataset['output']):
        prompt = generate_prompt(instruction, question)
        sequences = pipeline(
            prompt,
            max_length=140,
            do_sample=True,
            top_k=50,
            num_return_sequences=4,
            eos_token_id=tokenizer.eos_token_id,
        )
        responses = [seq['generated_text'] for seq in sequences]
        newResponses = []
        for response in responses:
            toFind = "### Response:"
            index = response.find(toFind)
            response = response.replace("??", "")
            response = response.replace("</s>", "")
            response = response.lower()
            newResponses.append(response[index+ len(toFind):].replace("??",""))
        labels.append(answer)
        preds.append(newResponses)



with open("/bigwork/nhwpshaa/pilot experiments/Predictions and Labels.csv", 'w', newline='') as writefile: #Prepare a CSV file with the candidate and reference sentences to compute automatic evaluation
    fieldnames = ['Predictions', 'Labels']
    writer = csv.DictWriter(writefile, fieldnames=fieldnames)
    writer.writeheader()
    for label,pred in zip(labels, preds):
        writer.writerow({'Predictions': pred, 'Labels': label})
