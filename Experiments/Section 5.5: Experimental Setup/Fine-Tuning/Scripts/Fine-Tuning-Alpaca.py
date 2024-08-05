import torch
import transformers
import sys
import os
import json
import os.path as osp
import pandas as pd
from datasets import Dataset,DatasetDict 
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, set_seed, EarlyStoppingCallback, IntervalStrategy, LlamaForCausalLM, LlamaTokenizer
from typing import List
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training, set_peft_model_state_dict
from typing import Union
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

def computeRouge(references, candidates):
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    return scores


def computeBleu1(prediction, label):
    bleuScore = 0
    prediction = prediction.replace("</s>", "")
    prediction = prediction.lower().split()
    label = label.lower().split()
    #print(f"Preds: {prediction} and labels: {label}")
    smoothingFunction = SmoothingFunction().method1
    return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(1,0,0,0), smoothing_function=smoothingFunction)


def computeBleu2(prediction, label):
    bleuScore = 0
    prediction = prediction.replace("</s>", "")
    prediction = prediction.lower().split()
    label = label.lower().split()
    #print(f"Preds: {prediction} and labels: {label}")
    smoothingFunction = SmoothingFunction().method1
    return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(0.5,0.5,0,0), smoothing_function=smoothingFunction)


def computeBleu3(prediction, label):
    bleuScore = 0
    prediction = prediction.replace("</s>", "")
    prediction = prediction.lower().split()
    label = label.lower().split()
    #print(f"Preds: {prediction} and labels: {label}")
    smoothingFunction = SmoothingFunction().method1
    return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(1/3,1/3,1/3), smoothing_function=smoothingFunction)


def computeBleu4(prediction, label):
    bleuScore = 0
    prediction = prediction.replace("</s>", "")
    prediction = prediction.lower().split()
    label = label.lower().split()
    #print(f"Preds: {prediction} and labels: {label}")
    smoothingFunction = SmoothingFunction().method1
    return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothingFunction)



os.environ["WANDB_DISABLED"] = "true" #disable weights and biases
set_seed(42) #set a seed to get reproducability
tokenizer = LlamaTokenizer.from_pretrained("/bigwork/nhwpshaa/alpaca-native") #load tokenizer

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with a question. Act as an expert in the domain from which the question stems and write a response that appropriately answers the question.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
    

class Prompter(object):
    __slots__ = ("template", "_verbose") #used to save memory. Here, only 2 attributes of the objects can be instantiated

    def __init__(self, name):
        with open(f"/Experiments/Section 5.5: Experimental Setup/Fine-Tuning/Training Dataset/{name}.json", 'r') as file:
            self.template = json.load(file)

    def generate_prompt(self,instruction,input, label) -> str:
        # returns the full prompt from instruction and input and if a label (=response, =output) is provided, it's also appended.
        response = self.template["prompt_input"].format(instruction=instruction, input=input)
        response = f"{response}{label}"
        return response

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    

def generateTestInstructions(dataset, trainDataset): #include examples from training dataset during test and validation to get better results 
    examples = ""
    for i,example in trainDataset.iterrows():
        examples += f"review: {example['Question']}\nResponse: {example['Answer']}\n"
    data = {"instruction":[], "input":[], "output":[]}
    for i, record in dataset.iterrows():
        question = record['Question']
        answer = record['Answer']
        if record['Question Type'].lower() == "yes/no question":
            data["instruction"].append(f"Answer the following questions with 'Yes' or 'No'")
        else:
            data["instruction"].append(f"Answer the following questions to the best of your knowledge. Answer the questions as briefly as possible using only one causal sentence. Here are some examples of questions and answers. Orient your answers to the examples"+examples)
        data["output"].append(answer)
        data["input"].append(question)
    return data


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


def generateAndTokenize(dataPoint): 
    prompt = prompter.generate_prompt(dataPoint["instruction"],dataPoint["input"],dataPoint["output"]    )
    tokenizedPrompt = tokenize(prompt)
    if not train_on_inputs: #why?
        user_prompt = prompter.generate_prompt(dataPpoint["instruction"], dataPoint["input"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]   #mask out labels
    return tokenizedPrompt


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id #if last token is not eos token 
        and len(result["input_ids"]) < cutoff_len  #if there are less values than the cutoff length
        and add_eos_token 
    ):
        result["input_ids"].append(tokenizer.eos_token_id) #add eos token 
        result["attention_mask"].append(1) #add attention mask
    result["labels"] = result["input_ids"].copy() #setting label equal to labels would enable the model to predict the next token based on the preceding token
    return result

                           
def loadSplits(prompting): #load the different taining dataset
    training = pd.read_csv("/Experiments/Section 5.5: Experimental Setup/Fine-Tuning/Training Dataset/First Order Effect and Cause/train.csv", sep=",", encoding="utf-8")
    validation = pd.read_csv("/Experiments/Section 5.5: Experimental Setup/Fine-Tuning/Training Dataset/First Order Effect and Cause/validation.csv", sep=",", encoding="utf-8")
    test= pd.read_csv("/Experiments/Section 5.5: Experimental Setup/Fine-Tuning/Training Dataset/First Order Effect and Cause/test.csv", sep=",", encoding="utf-8")
    print(f"Size before: {len(training)}")
    print(f"Size after: {len(training)}")
    dataset = {"train": training,"validation":validation , "test": test }
    for split in dataset:
        if prompting:
            if split != "train":
                dataset[split]= generateTestInstructions(dataset[split], dataset["train"])
        else:
            dataset[split] = generateTrainingInstructions(dataset[split])
    return dataset


generation_config = GenerationConfig()
prompter = Prompter("alpaca")
cutoff_len=1024
train_on_inputs=True
add_eos_token=True
prompt_template_name="alpaca"
lora_r = 8
lora_alpha = 16
lora_dropout = 0.01
lora_target_modules= ["q_proj", "v_proj", "k_proj"] #target the 3 modules

config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(model,
          data,
    batch_size: int = 256, #no. of samples taken from the dataset
    micro_batch_size: int = 4, #number of micro batch size, where value is the amount of samples per micro batch
    num_epochs: int = 30, #epoch = one loop of the whole training through the dataset
    learning_rate: float = 8e-4,    
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    #early_stopping_threshold = 0.001, #stops training when the difference in loss is less than 0.01 
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    ):
    gradientAccumulation = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Allow batched inference
    trainData = (data["train"].shuffle().map(generateAndTokenize))
    print(trainData)
    valData = (data["validation"].shuffle().map(generateAndTokenize))
    trainer = transformers.Trainer(
        model=model,
        train_dataset=trainData,
        eval_dataset=valData,
        #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradientAccumulation,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate, #rate at which the model's parameters are updated
            fp16=True, #accelerate training and reduce memory usage
            logging_steps=10,
            lr_scheduler_type='linear', #adjust the learning rate dynamically
            optim="adamw_torch",
            evaluation_strategy="steps", 
            metric_for_best_model="loss",
            save_strategy="steps",
            eval_steps=10,
            save_steps=10,
            save_total_limit=3,
            load_best_model_at_end = True,
            ddp_find_unused_parameters=None,
            group_by_length=False,
            output_dir="/Results/Adapter Cause and First-Order/",
            report_to=None,
        ), 
    )
    model.config.use_cache = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    trainer.train()
    model.save_pretrained("/Results/Adapter Cause and First-Order/")
    return model


def main(prompting=True, split="validation"):
    model_name = "/bigwork/nhwpshaa/alpaca-native"   #load the model from transformers or locally
    model = LlamaForCausalLM.from_pretrained(model_name, device_map={"":0}, load_in_8bit=True) 
    peft_model = get_peft_model(model, config) #load with peft to enable efficient fine-tuning
    dataset = loadSplits(prompting=prompting)
    data = DatasetDict()
    for k,v in dataset.items(): #copy all key and value pairs from the dataset
        data[k] = Dataset.from_dict(v)
    if prompting:        
        data[split] = (data[split].shuffle().map(generateAndTokenize))    
    if not prompting:
        learning_rate = 8e-4
        peft_model = train(peft_model,data, learning_rate=learning_rate) #train the model
    sample = data['test'] 
    test_dataloader = DataLoader(sample, batch_size=1)
    
    global preds, labels
    preds = []
    labels = []
    
    for step, (test_input) in enumerate(test_dataloader):
        prompt = generate_prompt(test_input["instruction"], test_input["input"]) #generate prompt from test dataset
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = peft_model.generate( #execute inference
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_length = 140
        )

        for s in generation_output.sequences: #for each output sequence
            output = tokenizer.decode(s) #deode the output
            #print(output)
            preds.append(output.split("### Response:")[1].strip())
        labels.append(test_input["output"])
    ## the code below is used to evaluate the performance of the model 
    cumulativeScore = 0
    cumulativeScore1 = 0
    cumulativeScore2 = 0
    cumulativeScore3 = 0
    cumulativeScore4 = 0
    cumulativeRouge1Precision = 0
    cumulativeRouge1Recall = 0
    cumulativeRouge1F1 = 0
    cumulativeRouge2Precision = 0
    cumulativeRouge2Recall = 0
    cumulativeRouge2F1 = 0
    cumulativeRougeLPrecision = 0
    cumulativeRougeLRecall = 0
    cumulativeRougeLF1 = 0
    for i in range(len(preds)):
        #print(f"This is the pred: {preds[i]} and this is the target: {labels[i]}")
        #print(f"This is the prediction: {preds[i]}")
        bleuScore = computeBleu(preds[i],labels[i][0])
        bleuScore1 = computeBleu1(preds[i],labels[i][0])
        bleuScore2 = computeBleu2(preds[i],labels[i][0])
        bleuScore3 = computeBleu3(preds[i],labels[i][0])
        bleuScore4 = computeBleu4(preds[i],labels[i][0])
        rougeScore = computeRouge(preds[i],labels[i][0])
        cumulativeScore +=  bleuScore
        cumulativeScore1 +=  bleuScore1
        cumulativeScore2 +=  bleuScore2
        cumulativeScore3 +=  bleuScore3
        cumulativeScore4 +=  bleuScore4
        cumulativeRouge1Precision += rougeScore['rouge-1']['p']
        cumulativeRouge1Recall = rougeScore['rouge-1']['r']
        cumulativeRouge1F1 = rougeScore['rouge-1']['f']
        cumulativeRouge2Precision = rougeScore['rouge-2']['p']
        cumulativeRouge2Recall = rougeScore['rouge-2']['r']
        cumulativeRouge2F1 = rougeScore['rouge-2']['f']
        cumulativeRougeLPrecision = rougeScore['rouge-l']['p']
        cumulativeRougeLRecall = rougeScore['rouge-l']['r']
        cumulativeRougeLF1 = rougeScore['rouge-l']['f']
        print(f"This is the BLEU score {bleuScore}")
        print(f"This is the BLEU-1 score {bleuScore1}")
        print(f"This is the BLEU-2 score {bleuScore2}")
        print(f"This is the BLEU-3 score {bleuScore3}")
        print(f"This is the BLEU-4 score {bleuScore4}")
        print(f"ROUGE-1: Precision: {rougeScore['rouge-1']['p']}, Recall: {rougeScore['rouge-1']['r']}, F1: {rougeScore['rouge-1']['f']}")
        print(f"ROUGE-2: Precision: {rougeScore['rouge-2']['p']}, Recall: {rougeScore['rouge-2']['r']}, F1: {rougeScore['rouge-2']['f']}")
        print(f"ROUGE-L: Precision: {rougeScore['rouge-l']['p']}, Recall: {rougeScore['rouge-l']['r']}, F1: {rougeScore['rouge-l']['f']}")
    avgScore = cumulativeScore / len(preds)
    avgScore1 = cumulativeScore1 / len(preds)
    avgScore2 = cumulativeScore2 / len(preds)
    avgScore3 = cumulativeScore3 / len(preds)
    avgScore4 = cumulativeScore4 / len(preds)
    avgRouge1Precision = cumulativeRouge1Precision / len(preds)
    avgRouge1Recall = cumulativeRouge1Recall / len(preds)
    avgRouge1F1 = cumulativeRouge1F1 / len(preds)
    avgRouge2Precision = cumulativeRouge2Precision / len(preds)
    avgRouge2Recall = cumulativeRouge2Recall / len(preds)
    avgRouge2F1 = cumulativeRouge2F1 / len(preds)
    avgRougeLPrecision = cumulativeRougeLPrecision / len(preds)
    avgRougeLRecall = cumulativeRougeLRecall / len(preds)
    avgRougeLF1 = cumulativeRougeLF1 / len(preds)
    print(f"Avg BLEU: {avgScore}")
    print(f"Avg BLEU-1: {avgScore1}")
    print(f"Avg BLEU-2: {avgScore2}")
    print(f"Avg BLEU-3: {avgScore3}")
    print(f"Avg BLEU-4: {avgScore4}")
    print(f"Avg Rouge1Precision: {avgRouge1Precision}")
    print(f"Avg Rouge1Recall: {avgRouge1Recall}")
    print(f"Avg Rouge1F1: {avgRouge1F1}")
    print(f"Avg Rouge2Precision: {avgRouge2Precision}")
    print(f"Avg Rouge2Recall: {avgRouge2Recall}")
    print(f"Avg Rouge2F1: {avgRouge2F1}")
    print(f"Avg RougeLPrecision: {avgRougeLPrecision}")
    print(f"Avg RougeLRecall: {avgRougeLRecall}")
    print(f"Avg RougeLF1: {avgRougeLF1}")
    

def computeBleu(prediction, label): #dynamic bleu score computation by adjusting the n-gram to the length of the input
    bleuScore = 0
    prediction = prediction.replace("</s>", "")
    prediction = prediction.lower().split()
    label = label.lower().split()
    #print(f"Preds: {prediction} and labels: {label}")
    smoothingFunction = SmoothingFunction().method1
    if len(label) == 1:
        return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(1,), smoothing_function=smoothingFunction)
    elif len(label) == 2:
        return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(0.5,0.5), smoothing_function=smoothingFunction)
    elif len(label) == 3:
        return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(1/3,1/3,1/3), smoothing_function=smoothingFunction)
    else:
        return nltk.translate.bleu_score.sentence_bleu([label], prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothingFunction)
        
        
acc = main(prompting=False, split="validation")
