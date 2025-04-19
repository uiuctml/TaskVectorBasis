import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, get_linear_schedule_with_warmup, get_constant_schedule
from datasets import load_dataset, Dataset
import evaluate
import wandb
import argparse
from datetime import datetime
import os
import random
import json
from datasets import Features, ClassLabel, Value
from torch.nn import CrossEntropyLoss
from transformers import pipeline
from collections import Counter
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import set_seed



OUTPUT_DIR = '/data/common/cindy2000_sh/tangent_task_arithmetic/fewshot_dataset'

class CustomDataCollatorForPrompting(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for example in examples:
            batch["input_ids"].append(example["input_ids"])
            batch["attention_mask"].append(example["attention_mask"])
            batch["labels"].append(example["labels"])

        batch = {k: torch.stack([torch.as_tensor(item) for item in v]) for k, v in batch.items()}
        return batch
    
class CustomTrainer(Trainer):
    def __init__(self, *args, label_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_map = label_map  

    def create_optimizer_and_scheduler(self, num_training_steps = 0):
        self.num_training_steps = num_training_steps
        if self.optimizer is None:
            self.select_trainable_parameters()
            self.no_decay = ["bias", "LayerNorm.weight"]
            self.init_opt(weight_decay=self.args.weight_decay, learning_rate=self.args.learning_rate)

    def select_trainable_parameters(self):
        params = {}
        for n, p in self.model.named_parameters():
            if 'encoder.layer' in n:
                layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                if layer_num >= self.args.fix_layers:
                    if not self.args.train_bias_only or 'bias' in n:
                        params[n] = p
                        print('Trainable: ', n)
                else:
                    print('Frozen: ', n)
            elif 'embeddings' in n:
                if not self.args.fix_embeddings:
                    params[n] = p
                    print('Trainable: ', n)
                else:
                    print('Frozen: ', n)
            else:
                if not self.args.fix_head:
                    if not self.args.train_bias_only or 'bias' in n:
                        params[n] = p
                        print('Trainable: ', n)
                else:
                    print('Frozen: ', n)
        self.params = params  #

    def init_opt(self, weight_decay, learning_rate):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.params.items() if not any(nd in n for nd in self.no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.params.items() if any(nd in n for nd in self.no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            if self.lr_scheduler is None:
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.num_training_steps
                )

        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=learning_rate,
            )
            if self.lr_scheduler is None:
                self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} is not implemented.")


class CustomTrainingArguments(TrainingArguments):
    def __init__(self, *args, fix_embeddings=False, fix_head=False, optimizer='AdamW', **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_embeddings = fix_embeddings
        self.fix_head = fix_head
        self.optimizer = optimizer
        self.fix_layers = 0
        self.train_bias_only = False

class CustomRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config, label_map=None):
        super().__init__(config)
        self.label_map = label_map

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        prediction_scores = outputs.logits  # [batch_size, seq_length, vocab_size]

        label_ids = list(self.label_map.values())
        filtered_prediction_scores = prediction_scores[:, :, label_ids]  # [batch_size, seq_length, len(label_ids)]
    
        if labels is not None:
            label_map_inverse = {v: i for i, v in enumerate(label_ids)}
            adjusted_labels = torch.tensor([label_map_inverse[label.item()] if label.item() in label_map_inverse else -100 for label in labels.view(-1)]).view(labels.size()).to(prediction_scores.device)
            
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(filtered_prediction_scores.view(-1, len(label_ids)), adjusted_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs[1:]  
        return outputs  


def create_few_shot_dataset(task_name, dataset, k):
    task_dir = os.path.join(OUTPUT_DIR, f"{task_name}-{k}-shot")
    
    if os.path.exists(task_dir):
        print(f"Loading existing {k}-shot dataset for {task_name} from {task_dir}")
        few_shot_dataset = Dataset.load_from_disk(task_dir)
        return few_shot_dataset
    
    random.seed(42)
    
    label_map = {}
    if task_name.startswith('anli'):
        for example in dataset[f"train_{task_name.split('_')[1]}"]:
            label = example["original_label"]
            if label not in label_map: 
                label_map[label] = []
            label_map[label].append(example)
    else:
        for example in dataset["train"]:
            label = example["original_label"]
            if label not in label_map: 
                label_map[label] = []
            label_map[label].append(example)

    few_shot_examples = []
    for label, examples in label_map.items():
        if len(examples) < k:
            sampled_examples = random.choices(examples, k=k)
        else:
            sampled_examples = random.sample(examples, k)  
        few_shot_examples.extend(sampled_examples)

    few_shot_dataset = Dataset.from_list(few_shot_examples)
    
    os.makedirs(task_dir, exist_ok=True)
    few_shot_dataset.save_to_disk(task_dir)
    print(f"Saved {k}-shot dataset for {task_name} at {task_dir}")

    return few_shot_dataset

def get_majority_label(train_dataset):
    random.seed(42)
    label_counts = Counter(train_dataset['original_label'])
    max_count = max(label_counts.values())
    majority_labels = [label for label, count in label_counts.items() if count == max_count]
    if len(majority_labels) > 1:
        return random.choice(majority_labels)
    else:
        return majority_labels[0]

def load_and_tokenize_data(dataset_name, tokenizer):

    if dataset_name == 'glue_qnli':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2 

            prompts = []
            for question, sentence in zip(examples["question"], examples["sentence"]):
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1: tokenizer.convert_tokens_to_ids('No')}

        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'glue_qnli_1sentence':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2 

            prompts = []
            for question, sentence in zip(examples["question"], examples["sentence"]):
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                prompt = f"{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1: tokenizer.convert_tokens_to_ids('No')}

        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'glue_mrpc':
        
        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2 

            prompts = []
            for question, sentence in zip(examples["sentence1"], examples["sentence2"]):
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)
            

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('No'),
                     1:  tokenizer.convert_tokens_to_ids('Yes') 
                    }
        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'glue_cola':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" This is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["sentence"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('incorrect'),
                     1:  tokenizer.convert_tokens_to_ids('correct')}
        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'glue_mnli':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premise"], examples["hypothesis"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1: tokenizer.convert_tokens_to_ids('Maybe'), 
                    2: tokenizer.convert_tokens_to_ids('No')}
        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation_matched']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'glue_qqp':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["question1"], examples["question2"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('No'),
                    1: tokenizer.convert_tokens_to_ids('Yes')}
        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'glue_rte':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["sentence1"], examples["sentence2"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1:  tokenizer.convert_tokens_to_ids('No'), 
                     }
        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'glue_sst2':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["sentence"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1:  tokenizer.convert_tokens_to_ids('great'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'glue_sst2_2sentence':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["sentence"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('nyu-mll/glue', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1:  tokenizer.convert_tokens_to_ids('great'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'sst5':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset("SetFit/sst5").rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenizer.add_tokens(['terrible','okay'])
        label_map = {0:  tokenizer.convert_tokens_to_ids('terrible'),
                     1:  tokenizer.convert_tokens_to_ids('bad'),
                     2:  tokenizer.convert_tokens_to_ids('okay'),
                     3:  tokenizer.convert_tokens_to_ids('good'),
                     4:  tokenizer.convert_tokens_to_ids('great'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'snli':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premise"], examples["hypothesis"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1:  tokenizer.convert_tokens_to_ids('Maybe'),
                     2: tokenizer.convert_tokens_to_ids('No')}
        dataset = load_dataset("stanfordnlp/snli").rename_column("label", "original_label").filter(lambda example: example["original_label"] != -1)
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'trec':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f"{tokenizer.mask_token}: "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('CogComp/trec', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("coarse_label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0:  tokenizer.convert_tokens_to_ids('Expression'), 
                     1: tokenizer.convert_tokens_to_ids('Entity'),
                     2:  tokenizer.convert_tokens_to_ids('Description'), 
                     3:  tokenizer.convert_tokens_to_ids('Human'), 
                     4:  tokenizer.convert_tokens_to_ids('Location'), 
                     5:  tokenizer.convert_tokens_to_ids('Number'), 
                     }
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'subj':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" This is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        features = Features({"label": ClassLabel(names=["subjective", "objective"]), 
                           "text": Value("string")})
        data_files = {
                        "train": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/subj/train.csv",
                        "test": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/subj/test.csv"
                     }
        dataset = load_dataset("csv", data_files=data_files, features=features, column_names=["label", "text"]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        tokenizer.add_tokens(['subjective','objective'])
        label_map = {0: tokenizer.convert_tokens_to_ids('subjective'), # both map to unknown
                     1: tokenizer.convert_tokens_to_ids('objective')}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'cr':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        features = Features({"label": ClassLabel(names=["negative", "positive"]),  
                           "text": Value("string")})
        data_files = {
                        "train": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/cr/train.csv",
                        "test": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/cr/test.csv"
                     }
        dataset = load_dataset("csv", data_files=data_files, features=features, column_names=["label", "text"]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1:  tokenizer.convert_tokens_to_ids('great')}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'mpqa':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        features = Features({"label": ClassLabel(names=["negative", "positive"]),  
                           "text": Value("string")})
        data_files = {
                        "train": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/mpqa/train.csv",
                        "test": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/mpqa/test.csv"
                     }
        dataset = load_dataset("csv", data_files=data_files, features=features, column_names=["label", "text"]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1:  tokenizer.convert_tokens_to_ids('great')}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'mr':
        
        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        features = Features({"label": ClassLabel(names=["negative", "positive"]),  
                           "text": Value("string")})
        data_files = {
                        "train": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/mr/train.csv",
                        "test": "/data/common/cindy2000_sh/tangent_task_arithmetic/local_dataset/mr/test.csv"
                     }
        dataset = load_dataset("csv", data_files=data_files, features=features, column_names=["label", "text"]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1: tokenizer.convert_tokens_to_ids('great')}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'ag_news':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This is about {tokenizer.mask_token} news."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        dataset = load_dataset('fancyzhx/ag_news').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('international'),
                     1: tokenizer.convert_tokens_to_ids('sports'),
                     2: tokenizer.convert_tokens_to_ids('business'),
                     3: tokenizer.convert_tokens_to_ids('science'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'yelp':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This place is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('yelp/yelp_review_full').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('poor'),
                     1: tokenizer.convert_tokens_to_ids('fair'),
                     2: tokenizer.convert_tokens_to_ids('good'),
                     3: tokenizer.convert_tokens_to_ids('great'),
                     4: tokenizer.convert_tokens_to_ids('excellent'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'imdb':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This movie is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('stanfordnlp/imdb').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1: tokenizer.convert_tokens_to_ids('great'),
                     }
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'yahoo_answer_topics':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f"This is related to {tokenizer.mask_token}. "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2

            prompts = []
            for question, content, sentence in zip(examples["question_title"], examples["question_content"], examples["best_answer"]):
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                content_ids = tokenizer.encode(content, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                total_length = len(question_ids) + len(content_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    extra_tokens = total_length - max_tokens_for_pair
                    sentence_truncation = extra_tokens
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_content = tokenizer.decode(content_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)

                prompt = f"{prompt_part}{truncated_question} {truncated_content} {truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('community-datasets/yahoo_answers_topics').rename_column("topic", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        tokenizer.add_tokens(['society','finance','entertainment','relationship'])
        label_map = {0: tokenizer.convert_tokens_to_ids('society'),
                     1: tokenizer.convert_tokens_to_ids('science'),
                     2: tokenizer.convert_tokens_to_ids('health'),
                     3: tokenizer.convert_tokens_to_ids('education'),
                     4: tokenizer.convert_tokens_to_ids('computer'),
                     5: tokenizer.convert_tokens_to_ids('sports'),
                     6: tokenizer.convert_tokens_to_ids('finance'),
                     7: tokenizer.convert_tokens_to_ids('entertainment'),
                     8: tokenizer.convert_tokens_to_ids('relationship'),
                     9: tokenizer.convert_tokens_to_ids('government'),
                     }
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name.startswith('anli'):
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premise"], examples["hypothesis"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1:  tokenizer.convert_tokens_to_ids('Maybe'),
                     2: tokenizer.convert_tokens_to_ids('No')}
        task_name = dataset_name.split("_")[1]
        dataset = load_dataset("facebook/anli").rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset[f'dev_{task_name}']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'glue_wnli':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["sentence1"], examples["sentence2"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0:  tokenizer.convert_tokens_to_ids('No'),
                     1: tokenizer.convert_tokens_to_ids('Yes')}
        task_name = dataset_name.split("_")
        dataset = load_dataset("nyu-mll/glue", 'wnli').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'superglue_boolq':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["passage"], examples["question"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_question}{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0:  tokenizer.convert_tokens_to_ids('No'),
                     1: tokenizer.convert_tokens_to_ids('Yes')}
        task_name = dataset_name.split("_")
        dataset = load_dataset("aps/super_glue", 'boolq', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'superglue_cb':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premise"], examples["hypothesis"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1:  tokenizer.convert_tokens_to_ids('No'),
                     2: tokenizer.convert_tokens_to_ids('Maybe')}
        dataset = load_dataset("aps/super_glue", 'cb', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'superglue_wic':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" Does  have the same meaning in both sentences? {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence, word in zip(examples["sentence1"], examples["sentence2"], examples["word"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids) + len(word)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_question}{truncated_sentence} Does {word} have the same meaning in both sentences? {tokenizer.mask_token}."
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0:  tokenizer.convert_tokens_to_ids('No'),
                     1: tokenizer.convert_tokens_to_ids('Yes')}
        dataset = load_dataset("aps/super_glue", 'wic', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'superglue_wsc':
        
        pos_tagger = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos")
        
        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" Does  refer to ?. {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence, word1, word2 in zip(examples["text"], examples["span1_text"], examples["span2_text"]):
                pos_tags1 = pos_tagger(word1)[0]["entity"]
                if pos_tags1.startswith("PRP"):
                    pronoun = word1
                    noun = word2
                else:
                    pronoun = word2
                    noun = word1
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence} Does {pronoun} refer to {noun}?. {tokenizer.mask_token}."
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        label_map = {0:  tokenizer.convert_tokens_to_ids('No'),
                     1: tokenizer.convert_tokens_to_ids('Yes')}
        dataset = load_dataset("aps/super_glue", 'wsc', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'ethics_commonsense':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["input"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('hendrycks/ethics', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('acceptable'),
                     1:  tokenizer.convert_tokens_to_ids('unacceptable'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'ethics_deontology':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"This is a {tokenizer.mask_token} excuse."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["scenario"], examples["excuse"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    sentence_truncation = extra_tokens
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_question}{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('great'),
                     1:  tokenizer.convert_tokens_to_ids('terrible') 
                    }
        dataset = load_dataset('hendrycks/ethics', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'ethics_justice':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["scenario"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('hendrycks/ethics', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('unfair'),
                     1:  tokenizer.convert_tokens_to_ids('fair'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'ethics_virtue':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["scenario"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('hendrycks/ethics', dataset_name.split('_')[1]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('bad'),
                     1:  tokenizer.convert_tokens_to_ids('good'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'logiqa2_nli':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f"? {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2 

            prompts = []
            for major, minor, conclusion in zip(examples["major_premise"], examples["minor_premise"], examples["conclusion"]):
                
                if major == []:
                    major_ids = tokenizer.encode([''], add_special_tokens=False)
                else:
                    major_ids = tokenizer.encode(major, add_special_tokens=False)
                minor_ids = tokenizer.encode(minor, add_special_tokens=False)
                conclusion_ids = tokenizer.encode(conclusion, add_special_tokens=False)

                
                total_length = len(major_ids) + len(minor_ids) + len(conclusion_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    major_truncation = max(1, extra_tokens//3)
                    minor_truncation = max(1, extra_tokens//3)
                    conclusion_truncation = extra_tokens - major_truncation - minor_truncation

                    
                    major_ids = major_ids[:, len(major_ids) - major_truncation]
                    minor_ids = minor_ids[:, len(minor_ids) - minor_truncation]
                    conclusion_ids = conclusion_ids[:, len(conclusion_ids) - conclusion_truncation]

                
                truncated_major = tokenizer.decode(major_ids, skip_special_tokens=True)
                truncated_minor = tokenizer.decode(minor_ids, skip_special_tokens=True).rstrip(".,;:!?")
                truncated_conclusion = tokenizer.decode(conclusion_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_major}{truncated_minor}{prompt_part}{truncated_conclusion}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}

        
        label_map = {0: tokenizer.convert_tokens_to_ids('No'),
                     1: tokenizer.convert_tokens_to_ids('Yes')}

        dataset = load_dataset('baber/logiqa2', 'logiqa2_nli', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'amazon':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This product is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('SetFit/amazon_reviews_multi_en').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('poor'),
                     1: tokenizer.convert_tokens_to_ids('fair'),
                     2: tokenizer.convert_tokens_to_ids('good'),
                     3: tokenizer.convert_tokens_to_ids('great'),
                     4: tokenizer.convert_tokens_to_ids('excellent'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'tweet_eval_emotion':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This person feels {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('cardiffnlp/tweet_eval', 'emotion').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenizer.add_tokens(['angry','optimistic', 'sad'])
        label_map = {0: tokenizer.convert_tokens_to_ids('angry'),
                     1: tokenizer.convert_tokens_to_ids('happy'),
                     2: tokenizer.convert_tokens_to_ids('optimistic'),
                     3: tokenizer.convert_tokens_to_ids('sad'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'tweet_eval_hate':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" The sentence is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('cardiffnlp/tweet_eval', 'hate').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('neutral'),
                     1: tokenizer.convert_tokens_to_ids('aggressive'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'tweet_eval_irony':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" The sentence is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('cardiffnlp/tweet_eval', 'irony').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenizer.add_tokens(['genuine','sarcastic'])
        label_map = {0: tokenizer.convert_tokens_to_ids('genuine'),
                     1: tokenizer.convert_tokens_to_ids('sarcastic'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'tweet_eval_offensive':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" It is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('cardiffnlp/tweet_eval', 'offensive').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('polite'),
                     1: tokenizer.convert_tokens_to_ids('offensive'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'tweet_eval_sentiment':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('cardiffnlp/tweet_eval', 'sentiment').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenizer.add_tokens(['terrible'])
        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1: tokenizer.convert_tokens_to_ids('okay'),
                     2: tokenizer.convert_tokens_to_ids('great'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'rotten_tomatoes':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This is {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('terrible'),
                     1: tokenizer.convert_tokens_to_ids('great'),}
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'dbpedia_14':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This is about {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["content"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        dataset = load_dataset('fancyzhx/dbpedia_14').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('company'),
                     1: tokenizer.convert_tokens_to_ids('school'),
                     2: tokenizer.convert_tokens_to_ids('artist'),
                     3: tokenizer.convert_tokens_to_ids('sports'),
                     4: tokenizer.convert_tokens_to_ids('politics'),
                     5: tokenizer.convert_tokens_to_ids('transportation'),
                     6: tokenizer.convert_tokens_to_ids('building'),
                     7: tokenizer.convert_tokens_to_ids('nature'),
                     8: tokenizer.convert_tokens_to_ids('town'),
                     9: tokenizer.convert_tokens_to_ids('animal'),
                     10: tokenizer.convert_tokens_to_ids('plant'),
                     11: tokenizer.convert_tokens_to_ids('music'),
                     12: tokenizer.convert_tokens_to_ids('movie'),
                     13: tokenizer.convert_tokens_to_ids('book'),}

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'emotion':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This person feels {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('dair-ai/emotion', 'split').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenizer.add_tokens(['sad'])
        label_map = {0: tokenizer.convert_tokens_to_ids('happy'),
                     1: tokenizer.convert_tokens_to_ids('sad'),
                     2: tokenizer.convert_tokens_to_ids('anger'),
                     3: tokenizer.convert_tokens_to_ids('scared'),
                     4: tokenizer.convert_tokens_to_ids('love'),
                     5: tokenizer.convert_tokens_to_ids('shock'),}

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
   
    elif dataset_name == 'paws':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["sentence1"], examples["sentence2"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('No'),
                    1: tokenizer.convert_tokens_to_ids('Yes')}
        dataset = load_dataset('google-research-datasets/paws', 'labeled_final').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == '20_newsgroups':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This is about {tokenizer.mask_token} news."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        dataset = load_dataset('SetFit/20_newsgroups').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']
        tokenizer.add_tokens(['atheism','graphics','ibm','windowsX','purchase','motorcycle','baseball','hockey',
                              'cryptography','electronics','christian','mideast','religion'])
        label_map = {0: tokenizer.convert_tokens_to_ids('atheism'),
                     1: tokenizer.convert_tokens_to_ids('graphics'), 
                     2: tokenizer.convert_tokens_to_ids('windows'),
                     3: tokenizer.convert_tokens_to_ids('ibm'),
                     4: tokenizer.convert_tokens_to_ids('mac'),
                     5: tokenizer.convert_tokens_to_ids('windowsX'),
                     6: tokenizer.convert_tokens_to_ids('purchase'),
                     7: tokenizer.convert_tokens_to_ids('car'),
                     8: tokenizer.convert_tokens_to_ids('motorcycle'),
                     9: tokenizer.convert_tokens_to_ids('baseball'),
                     10: tokenizer.convert_tokens_to_ids('hockey'),
                     11: tokenizer.convert_tokens_to_ids('cryptography'),
                     12: tokenizer.convert_tokens_to_ids('electronics'),
                     13: tokenizer.convert_tokens_to_ids('health'),
                     14: tokenizer.convert_tokens_to_ids('space'),
                     15: tokenizer.convert_tokens_to_ids('christian'),
                     16: tokenizer.convert_tokens_to_ids('gun'),
                     17: tokenizer.convert_tokens_to_ids('mideast'),
                     18: tokenizer.convert_tokens_to_ids('politics'),
                     19: tokenizer.convert_tokens_to_ids('religion'),
                     }
        
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'folio':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premises"], examples["conclusion"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}


        label2int = {
                        "True": 0,
                        "Uncertain": 1,
                        "False": 2
                    }

        
        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1: tokenizer.convert_tokens_to_ids('Maybe'),
                     2: tokenizer.convert_tokens_to_ids('No')}

        dataset = load_dataset('tasksource/folio').rename_column("label", "original_label")
        dataset = dataset.map(lambda example: {"original_label": label2int[example["original_label"]]})
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'doc-nli':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premise"], examples["hypothesis"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}


        label2int = {
                        "entailment": 0,
                        "not_entailment": 1,
                    }

        
        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1: tokenizer.convert_tokens_to_ids('No')}

        dataset = load_dataset('tasksource/doc-nli').rename_column("label", "original_label")
        dataset = dataset.map(lambda example: {"original_label": label2int[example["original_label"]]})
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'wanli':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premise"], examples["hypothesis"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}


        label2int = {
                        "neutral": 0,
                        "entailment": 1,
                        "contradiction": 2,
                    }

        
        label_map = {0: tokenizer.convert_tokens_to_ids('Maybe'),
                     1: tokenizer.convert_tokens_to_ids('Yes'),
                     2: tokenizer.convert_tokens_to_ids('No')}

        dataset = load_dataset('alisawuffles/WANLI').rename_column("label", "original_label")
        dataset = dataset.map(lambda example: {"original_label": label2int[example["original_label"]]})
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'vitaminc':

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["evidence"], examples["claim"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                truncated_sentence = truncated_sentence[0].lower() + truncated_sentence[1:]  

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}


        label2int = {
                        "SUPPORTS": 0,
                        "NOT ENOUGH INFO": 1,
                        "REFUTES": 2,
                    }

        
        label_map = {0: tokenizer.convert_tokens_to_ids('Yes'),
                     1: tokenizer.convert_tokens_to_ids('Maybe'),
                     2: tokenizer.convert_tokens_to_ids('No')}

        dataset = load_dataset('tals/vitaminc').rename_column("label", "original_label")
        dataset = dataset.map(lambda example: {"original_label": label2int[example["original_label"]]})
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name.startswith('babi_nli'):

        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" {tokenizer.mask_token}, "
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence in zip(examples["premise"], examples["hypothesis"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_question}{prompt_part}{truncated_sentence}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}


        
        label_map = {0: tokenizer.convert_tokens_to_ids('No'),
                     1: tokenizer.convert_tokens_to_ids('Yes')}

        dataset = load_dataset('tasksource/babi_nli', dataset_name.split('_')[2]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'fake_news':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was {tokenizer.mask_token} news."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('GonzaloA/fake_news').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('fake'),
                     1:  tokenizer.convert_tokens_to_ids('real'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name.startswith('human-vs-machine'):

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was written by {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('NicolaiSivesind/human-vs-machine', dataset_name[17:]).rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        label_map = {0: tokenizer.convert_tokens_to_ids('human'),
                     1:  tokenizer.convert_tokens_to_ids('machine'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'AI-human-text':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" It was written by {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["text"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}


        dataset = load_dataset('andythetechnerd03/AI-human-text').rename_column("generated", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        label_map = {0: tokenizer.convert_tokens_to_ids('human'),
                     1:  tokenizer.convert_tokens_to_ids('machine'),}
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'wos':

        def tokenize_function(examples, tokenizer, label_map):
            prompt_part = f" This is about {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_sentence = args.max_sequence_length - prompt_length - 2  
            prompts = []
            for sentence in examples["abstract"]:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_tokens_for_sentence)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)
                prompt = f"{truncated_sentence}{prompt_part}"
                prompts.append(prompt)
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}
        
        label2int = {
                        "Medical": 0,
                        "Psychology": 1,
                        "CS": 2,
                        "biochemistry": 3,
                        "ECE": 4,
                        "Civil": 5,
                        "MAE": 6,
                    }

        dataset = load_dataset('river-martin/web-of-science-with-label-texts').rename_column("domain", "original_label")
        dataset = dataset.map(lambda example: {"original_label": label2int[example["original_label"]]})
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['test']

        tokenizer.add_tokens(['Medicine','Psychology','CS','Biochemistry','ECE','CivilE','MaterialE'])
        label_map = {0: tokenizer.convert_tokens_to_ids('Medicine'),
                     1: tokenizer.convert_tokens_to_ids('Psychology'),
                     2: tokenizer.convert_tokens_to_ids('CS'),
                     3: tokenizer.convert_tokens_to_ids('Biochemistry'),
                     4: tokenizer.convert_tokens_to_ids('ECE'),
                     5: tokenizer.convert_tokens_to_ids('CivilE'),
                     6: tokenizer.convert_tokens_to_ids('MaterialE'),
                     }
        
       
        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'phrase_similarity':
        
        def tokenize_function(examples, tokenizer, label_map):
           
            
            prompt_part = f" Does  and   have the same meaning? {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

           
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for question, sentence, word1, word2 in zip(examples["sentence1"], examples["sentence2"], examples["phrase1"], examples["phrase2"]):
                
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)

                
                total_length = len(question_ids) + len(sentence_ids) + len(word1) + len(word2)
                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    question_truncation = max(1, extra_tokens // 2)
                    sentence_truncation = extra_tokens - question_truncation

                    
                    question_ids = question_ids[: len(question_ids) - question_truncation]
                    sentence_ids = sentence_ids[: len(sentence_ids) - sentence_truncation]

                
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_sentence = tokenizer.decode(sentence_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_question}{truncated_sentence} Does {word1} and {word2} have the same meaning? {tokenizer.mask_token}."
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
    
            return {k: v for k, v in inputs.items()}

        label_map = {0:  tokenizer.convert_tokens_to_ids('No'),
                     1: tokenizer.convert_tokens_to_ids('Yes')}
        dataset = load_dataset("PiC/phrase_similarity").rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'art':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" Question: which hypothesis is correct? Answer: {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2

            prompts = []
            for obs1, obs2, hyp1, hyp2 in zip(examples["observation_1"], examples["observation_2"], examples["hypothesis_1"], examples["hypothesis_2"]):
                obs1_ids = tokenizer.encode(obs1, add_special_tokens=False)
                obs2_ids = tokenizer.encode(obs2, add_special_tokens=False)
                hyp1_ids = tokenizer.encode(f"A: {hyp1}", add_special_tokens=False)
                hyp2_ids = tokenizer.encode(f"B: {hyp2}", add_special_tokens=False)

                
                total_length = len(obs1_ids) + len(obs2_ids) + len(hyp1_ids) + len(hyp2_ids)
                if total_length > max_tokens_for_pair:
                    extra_tokens = total_length - max_tokens_for_pair
                    truncation_per_section = max(1, extra_tokens // 4)

                    obs1_ids = obs1_ids[:-truncation_per_section] if len(obs1_ids) > truncation_per_section else obs1_ids
                    obs2_ids = obs2_ids[:-truncation_per_section] if len(obs2_ids) > truncation_per_section else obs2_ids
                    hyp1_ids = hyp1_ids[:-truncation_per_section] if len(hyp1_ids) > truncation_per_section else hyp1_ids
                    hyp2_ids = hyp2_ids[:-truncation_per_section] if len(hyp2_ids) > truncation_per_section else hyp2_ids

                truncated_obs1 = tokenizer.decode(obs1_ids, skip_special_tokens=True)
                truncated_obs2 = tokenizer.decode(obs2_ids, skip_special_tokens=True)
                truncated_hyp1 = tokenizer.decode(hyp1_ids, skip_special_tokens=True)
                truncated_hyp2 = tokenizer.decode(hyp2_ids, skip_special_tokens=True)

                prompt = f"{truncated_obs1} {truncated_obs2} {truncated_hyp1} {truncated_hyp2}{prompt_part}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]
            
            return {k: v for k, v in inputs.items()}
        label2int = {
                        1: 0,
                        2: 1
                    }
        
        label_map = {0: tokenizer.convert_tokens_to_ids('A'),
                     1: tokenizer.convert_tokens_to_ids('B')}

        dataset = load_dataset('allenai/art').rename_column("label", "original_label")
        dataset = dataset.map(lambda example: {"original_label": label2int[example["original_label"]]})
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'ARC-Easy':

        def tokenize_function(examples, tokenizer, label_map):
            prompts = []
            for question, choices, answer_key in zip(examples["question"], examples["choices"], examples["original_label"]):
                
                prompt = f"{question}"

                
                for idx, choice_text in enumerate(choices["text"]):
                    choice_letter = chr(65 + idx)  
                    prompt += f" {choice_letter}: {choice_text}"

                prompt += f" Question: which is the best answer? Answer: {tokenizer.mask_token}."

                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            def convert_to_numeric(label):
                if isinstance(label, str) and label.isalpha():  
                    return ord(label.upper()) - 65 
                else:
                    return int(label)

            labels = [label_map[convert_to_numeric(answer)] for answer in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('A'),
                     1: tokenizer.convert_tokens_to_ids('B'),
                     2: tokenizer.convert_tokens_to_ids('C'),
                     3: tokenizer.convert_tokens_to_ids('D'),
                     4: tokenizer.convert_tokens_to_ids('E'),
                     }

        dataset = load_dataset('allenai/ai2_arc', dataset_name).rename_column("answerKey", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'ARC-Challenge':

        def tokenize_function(examples, tokenizer, label_map):
            prompts = []
            for question, choices, answer_key in zip(examples["question"], examples["choices"], examples["original_label"]):
                
                prompt = f"{question}"

                
                for idx, choice_text in enumerate(choices["text"]):
                    choice_letter = chr(65 + idx)  
                    prompt += f" {choice_letter}: {choice_text}"

                prompt += f" Question: which is the best answer? Answer: {tokenizer.mask_token}."

                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            def convert_to_numeric(label):
                if isinstance(label, str) and label.isalpha():  
                    return ord(label.upper()) - 65  
                else:
                    return int(label)


            labels = [label_map[convert_to_numeric(answer)] for answer in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids('A'),
                     1: tokenizer.convert_tokens_to_ids('B'),
                     2: tokenizer.convert_tokens_to_ids('C'),
                     3: tokenizer.convert_tokens_to_ids('D'),
                     4: tokenizer.convert_tokens_to_ids('E'),
                     }


        dataset = load_dataset('allenai/ai2_arc', dataset_name).rename_column("answerKey", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'superglue_copa':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" Answer: {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for premise, question, choice1, choice2 in zip(examples["premise"], examples["question"], examples["choice1"], examples["choice2"]):
                
                premise_ids = tokenizer.encode(premise, add_special_tokens=False)
                question_ids = tokenizer.encode(f"Question: {question}", add_special_tokens=False)
                choice1_ids = tokenizer.encode(f"A: {choice1}", add_special_tokens=False)
                choice2_ids = tokenizer.encode(f"B: {choice2}", add_special_tokens=False)

                
                total_length = len(premise_ids) + len(question_ids) + len(choice1_ids) + len(choice2_ids)
                num_sections = 4  

                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    truncation_per_section = max(1, extra_tokens // num_sections)

                    
                    premise_ids = premise_ids[:-truncation_per_section] if len(premise_ids) > truncation_per_section else premise_ids
                    question_ids = question_ids[:-truncation_per_section] if len(question_ids) > truncation_per_section else question_ids
                    choice1_ids = choice1_ids[:-truncation_per_section] if len(choice1_ids) > truncation_per_section else choice1_ids
                    choice2_ids = choice2_ids[:-truncation_per_section] if len(choice2_ids) > truncation_per_section else choice2_ids

                
                truncated_premise = tokenizer.decode(premise_ids, skip_special_tokens=True)
                truncated_question = tokenizer.decode(question_ids, skip_special_tokens=True)
                truncated_choice1 = tokenizer.decode(choice1_ids, skip_special_tokens=True)
                truncated_choice2 = tokenizer.decode(choice2_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_premise} {truncated_question} {truncated_choice1} {truncated_choice2}{prompt_part}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids("A"),
                    1: tokenizer.convert_tokens_to_ids("B")}

        dataset = load_dataset("super_glue", 'copa', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'hellaswag':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" Question: which is the best ending? Answer: {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for ctx, endings in zip(examples["ctx"], examples["endings"]):
                
                ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
                ending_ids = [tokenizer.encode(f"{chr(65 + idx)}: {ending}", add_special_tokens=False) for idx, ending in enumerate(endings)]

                
                total_length = len(ctx_ids) + sum(len(e) for e in ending_ids)
                num_sections = len(ending_ids) + 1  

                if total_length > max_tokens_for_pair:
                    extra_tokens = total_length - max_tokens_for_pair
                    truncation_per_section = max(1, extra_tokens // num_sections)

                    
                    ctx_ids = ctx_ids[:-truncation_per_section] if len(ctx_ids) > truncation_per_section else ctx_ids
                    ending_ids = [e[:-truncation_per_section] if len(e) > truncation_per_section else e for e in ending_ids]

                
                truncated_ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)
                truncated_endings = [tokenizer.decode(e, skip_special_tokens=True) for e in ending_ids]

                
                prompt = f"{truncated_ctx} {truncated_endings[0]} {truncated_endings[1]} {truncated_endings[2]} {truncated_endings[3]}{prompt_part}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            
            labels = [label_map[int(label)] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        
        label_map = {0: tokenizer.convert_tokens_to_ids("A"),
                     1: tokenizer.convert_tokens_to_ids("B"),
                     2: tokenizer.convert_tokens_to_ids("C"),
                     3: tokenizer.convert_tokens_to_ids("D")}

        dataset = load_dataset('Rowan/hellaswag').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'piqa':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" Question: which is the best solution? Answer: {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            
            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for goal, sol1, sol2 in zip(examples["goal"], examples["sol1"], examples["sol2"]):
                
                goal_ids = tokenizer.encode(goal, add_special_tokens=False)
                sol1_ids = tokenizer.encode(f"A: {sol1}", add_special_tokens=False)
                sol2_ids = tokenizer.encode(f"B: {sol2}", add_special_tokens=False)

                
                total_length = len(goal_ids) + len(sol1_ids) + len(sol2_ids)
                num_sections = 3  

                if total_length > max_tokens_for_pair:
                    extra_tokens = total_length - max_tokens_for_pair
                    truncation_per_section = max(1, extra_tokens // num_sections)

                    
                    goal_ids = goal_ids[:-truncation_per_section] if len(goal_ids) > truncation_per_section else goal_ids
                    sol1_ids = sol1_ids[:-truncation_per_section] if len(sol1_ids) > truncation_per_section else sol1_ids
                    sol2_ids = sol2_ids[:-truncation_per_section] if len(sol2_ids) > truncation_per_section else sol2_ids

                
                truncated_goal = tokenizer.decode(goal_ids, skip_special_tokens=True)
                truncated_sol1 = tokenizer.decode(sol1_ids, skip_special_tokens=True)
                truncated_sol2 = tokenizer.decode(sol2_ids, skip_special_tokens=True)

                
                prompt = f"{truncated_goal} {truncated_sol1} {truncated_sol2}{prompt_part}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids("A"),
                    1: tokenizer.convert_tokens_to_ids("B")}

        dataset = load_dataset('ybisk/piqa', cache_dir='/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name == 'swag':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = f" Question: which is the best answer? Answer: {tokenizer.mask_token}."
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            max_tokens_for_pair = args.max_sequence_length - prompt_length - 2  

            prompts = []
            for startphrase, ending0, ending1, ending2, ending3 in zip(
                examples["startphrase"], examples["ending0"], examples["ending1"], examples["ending2"], examples["ending3"]
            ):
                
                startphrase_ids = tokenizer.encode(startphrase, add_special_tokens=False)
                ending_ids = [
                    tokenizer.encode(f"{chr(65 + idx)}: {ending}", add_special_tokens=False)
                    for idx, ending in enumerate([ending0, ending1, ending2, ending3])
                ]

                
                total_length = len(startphrase_ids) + sum(len(e) for e in ending_ids)
                num_sections = len(ending_ids) + 1  

                if total_length > max_tokens_for_pair:
                    
                    extra_tokens = total_length - max_tokens_for_pair
                    truncation_per_section = max(1, extra_tokens // num_sections)

                    
                    startphrase_ids = startphrase_ids[:-truncation_per_section] if len(startphrase_ids) > truncation_per_section else startphrase_ids
                    ending_ids = [e[:-truncation_per_section] if len(e) > truncation_per_section else e for e in ending_ids]

                
                truncated_startphrase = tokenizer.decode(startphrase_ids, skip_special_tokens=True)
                truncated_endings = [tokenizer.decode(e, skip_special_tokens=True) for e in ending_ids]

                
                prompt = f"{truncated_startphrase} {truncated_endings[0]} {truncated_endings[1]} {truncated_endings[2]} {truncated_endings[3]}{prompt_part}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            
            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        
        label_map = {0: tokenizer.convert_tokens_to_ids("A"),
                    1: tokenizer.convert_tokens_to_ids("B"),
                    2: tokenizer.convert_tokens_to_ids("C"),
                    3: tokenizer.convert_tokens_to_ids("D")}

        dataset = load_dataset('allenai/swag', 'regular').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
    
    elif dataset_name == 'siqa':

        def tokenize_function(examples, tokenizer, label_map):
            
            prompt_part = " Answer: " + tokenizer.mask_token
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            prompts = []
            for context, question, answerA, answerB, answerC in zip(
                examples["context"], examples["question"], examples["answerA"], examples["answerB"], examples["answerC"]
            ):
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                
                max_tokens_for_context = args.max_sequence_length - prompt_length - 2  

                if len(context_ids) > max_tokens_for_context:
                    context_ids = context_ids[:max_tokens_for_context]

                truncated_context = tokenizer.decode(context_ids, skip_special_tokens=True)
                
                prompt = f"{truncated_context} Question: {question} A: {answerA}, B: {answerB}, C: {answerC}.{prompt_part}"
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            labels = [label_map[int(label)] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        label_map = {0: tokenizer.convert_tokens_to_ids("A"),
                    1: tokenizer.convert_tokens_to_ids("B"),
                    2: tokenizer.convert_tokens_to_ids("C"),
                    3: tokenizer.convert_tokens_to_ids("D")}

        dataset = load_dataset('allenai/social_i_qa', cache_dir = '/home/cindy2000_sh/.cache/huggingface').rename_column("label", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    elif dataset_name.startswith('race'):

        def tokenize_function(examples, tokenizer, label_map):
            examples["original_label"] = [ord(ans) - ord("A") for ans in examples["original_label"]]

            
            prompt_part = " Answer: " + tokenizer.mask_token
            prompt_length = len(tokenizer(prompt_part, add_special_tokens=False)["input_ids"])

            prompts = []
            for article, question, options in zip(examples["article"], examples["question"], examples["options"]):
                article_ids = tokenizer.encode(article, add_special_tokens=False)
                
                max_tokens_for_article = args.max_sequence_length - prompt_length - 2  

                if len(article_ids) > max_tokens_for_article:
                    article_ids = article_ids[:max_tokens_for_article]

                truncated_article = tokenizer.decode(article_ids, skip_special_tokens=True)

                prompt = (f"{truncated_article} Question: {question} "
                        f"A: {options[0]}, B: {options[1]}, C: {options[2]}, D: {options[3]}.{prompt_part}")
                prompts.append(prompt)

            
            inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_sequence_length)

            
            inputs["labels"] = torch.full_like(inputs["input_ids"], -100)

            
            mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)

            
            labels = [label_map[label] for label in examples["original_label"]]

            
            for i, mask_idx in enumerate(mask_indices[0].unique()):
                inputs["labels"][mask_idx, mask_indices[1][i]] = labels[i]

            return {k: v for k, v in inputs.items()}

        
        label_map = {0: tokenizer.convert_tokens_to_ids("A"),
                    1: tokenizer.convert_tokens_to_ids("B"),
                    2: tokenizer.convert_tokens_to_ids("C"),
                    3: tokenizer.convert_tokens_to_ids("D")}

        dataset = load_dataset('ehovy/race', dataset_name.split('_')[1])
        dataset = dataset.rename_column("answer", "original_label")
        train_dataset = create_few_shot_dataset(dataset_name, dataset, args.k)
        validation_dataset = dataset['validation']

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)
        tokenized_validation = validation_dataset.map(lambda x: tokenize_function(x, tokenizer, label_map), batched=True)

    else:
        raise NotImplementedError

    return tokenized_train, tokenized_validation, label_map, tokenizer

def preprocess_logits_for_metrics(logits, labels):
    # hack: label_map global
    label_ids = list(label_map.values())
    filtered_logits = logits[:, :, label_ids] 
    pred_ids = torch.argmax(filtered_logits, dim=-1)
    return pred_ids


def compute_metrics(eval_pred):
    # hack: label_map global
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    
    mask = labels != -100
    labels = labels[mask]
    token_id_to_label = {v: k for k, v in label_map.items()}
    labels = [token_id_to_label[label.item()] for label in labels]
    predictions = predictions[mask]
     
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

def is_valid_label_map(label_map):
    token_ids = list(label_map.values())
    unique_token_ids = len(token_ids) == len(set(token_ids))
    
    keys = list(label_map.keys())
    sorted_and_consecutive_keys = keys == list(range(len(keys)))

    return unique_token_ids and sorted_and_consecutive_keys

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def train_pipeline():
    set_random_seed(args.seed)
    num_classes = len(label_map)
    model = CustomRobertaForMaskedLM.from_pretrained(model_name, label_map=label_map)
    model.resize_token_embeddings(len(tokenizer))

    training_args = CustomTrainingArguments(
        output_dir=f"/data-4/common/cindy2000_sh/tangent_task_arithmetic/{model_name}/{args.dataset}/{args.seed}",
        evaluation_strategy="steps",  
        eval_steps= 4 * args.k * num_classes,  
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps = 16 * args.k * num_classes, 
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps", 
        save_steps= 16 * args.k * num_classes,  
        save_total_limit = 1,
        load_best_model_at_end=True,  
        metric_for_best_model="accuracy", 
        greater_is_better=True,  
        report_to="wandb",
        fix_embeddings=True,
        fix_head=True,
        optimizer='AdamW',
    )

    data_collator = CustomDataCollatorForPrompting(tokenizer=tokenizer, mlm=True)
    

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = validation_dataset,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    wandb.init(project="task_vector_basis", entity="structured_task", name=datetime.now().strftime("%m%d%Y%H%M%S"))
    wandb.config.dataset = args.dataset

    trainer.train()

    if training_args.load_best_model_at_end:
        eval_results = trainer.evaluate()  
        print("Evaluation results for the best model:", eval_results)

        results_path = os.path.join(training_args.output_dir, "best_model_eval_results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f)
        print(f"Best model evaluation results saved at {results_path}")

        best_model_path = os.path.join(training_args.output_dir, "best_model")
        trainer.save_model(best_model_path)
        print(f"Best model saved at {best_model_path}")


def eval_majority_pipeline():
    base_dir = f"/data-4/common/cindy2000_sh/tangent_task_arithmetic/{model_name}"
    dataset_name = args.dataset

    results = {
        "dataset_name": [],
        "fine_tune_accuracy": [],
        "majority_label_accuracy": []
    }

    for dataset_name in tqdm(os.listdir(base_dir)):
        dataset_path = os.path.join(base_dir, dataset_name)
        if os.path.isdir(dataset_path):
            train_dataset, validation_dataset, _, _ = load_and_tokenize_data(dataset_name, tokenizer)
            eval_results_path = os.path.join(dataset_path, "best_model_eval_results.json")
            if os.path.exists(eval_results_path):
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)
                fine_tune_accuracy = eval_results.get("eval_accuracy", None)
            else:
                fine_tune_accuracy = None

            train_majority_label = get_majority_label(train_dataset)
            test_labels = validation_dataset['original_label']  
            predictions = [train_majority_label] * len(test_labels)  
            majority_accuracy = accuracy_score(test_labels, predictions)
            majority_accuracy_path = os.path.join(dataset_path, "majority_accuracy.json")
            with open(majority_accuracy_path, "w") as f:
                json.dump({"majority_accuracy": majority_accuracy}, f)
            
            results["dataset_name"].append(dataset_name)
            results["fine_tune_accuracy"].append(fine_tune_accuracy)
            results["majority_label_accuracy"].append(majority_accuracy)

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(base_dir, f"accuracy_comparison_{args.seed}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Accuracy comparison saved at {csv_path}")

def select_roberta_parameters(model):
    params = {}
    for n, p in model.named_parameters():
        if n.startswith("roberta.encoder.layer"):
            params[n] = p
    return params

def load_finetuned_checkpoint(checkpoint_path):
    model = RobertaForMaskedLM.from_pretrained(checkpoint_path)
    encoder_params = select_roberta_parameters(model)
    return encoder_params

def load_task_vector(task_path, pretrained_model):
    finetuned_params = load_finetuned_checkpoint(task_path)
    return create_task_vector(pretrained_model, finetuned_params) 

def create_task_vector(pretrained_model, finetuned_params):
    pretrained_params = select_roberta_parameters(pretrained_model)
    task_vector = {}
    for name, param_pretrained in pretrained_params.items():
        if name in finetuned_params:
            task_vector[name] = finetuned_params[name].data - param_pretrained.data
    return task_vector

def get_pretrained_model():
    pretrained_checkpoint_path = f"/data/common/cindy2000_sh/tangent_task_arithmetic/{model_name}/pretrained_backbone_checkpoint.pt"
    if os.path.exists(pretrained_checkpoint_path):
        print("Loading pretrained backbone checkpoint.")
        return torch.load(pretrained_checkpoint_path)
    else:
        print("Saving pretrained backbone checkpoint.")
        model = RobertaForMaskedLM.from_pretrained(model_name) 
        torch.save(model, pretrained_checkpoint_path)
        return model

def recover_model_from_task_vector(task_vector, pretrained_model):
    model_state_dict = pretrained_model.state_dict()

    for name, delta in task_vector.items():
        if name in model_state_dict:
            model_state_dict[name] += delta
        else:
            print(f"Warning: {name} not found in model parameters; skipping.")

    pretrained_model.load_state_dict(model_state_dict)
    return pretrained_model




def ties_merging(flat_task_checks, reset_thresh=None, merge_func="dis-mean"):
    
    def topk_values_mask(M, K, return_mask=False):
        if K > 1:
            K /= 100

        original_shape = M.shape
        if M.dim() == 1:
            M = M.unsqueeze(0)

        n, d = M.shape
        k = int(d * K)
        k = d - k  
        kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
        mask = M.abs() >= kth_values
        final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

        if return_mask:
            return M * final_mask, final_mask.float().mean(dim=1), final_mask
        return M * final_mask, final_mask.float().mean(dim=1)

    def resolve_zero_signs(sign_to_mult, method="majority"):
        majority_sign = torch.sign(sign_to_mult.sum())

        if method == "majority":
            sign_to_mult[sign_to_mult == 0] = majority_sign
        elif method == "minority":
            sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
        return sign_to_mult

    def resolve_sign(Tensor):
        sign_to_mult = torch.sign(Tensor.sum(dim=0))
        sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
        return sign_to_mult

    def disjoint_merge(Tensor, merge_func, sign_to_mult):
        merge_func = merge_func.split("-")[-1]

        if sign_to_mult is not None:
            rows_to_keep = torch.where(
                sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
            )
            selected_entries = Tensor * rows_to_keep
        else:
            rows_to_keep = Tensor != 0
            selected_entries = Tensor * rows_to_keep

        if merge_func == "mean":
            non_zero_counts = (selected_entries != 0).sum(dim=0).float()
            disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
                non_zero_counts, min=1
            )
        elif merge_func == "sum":
            disjoint_aggs = torch.sum(selected_entries, dim=0)
        elif merge_func == "max":
            disjoint_aggs = selected_entries.abs().max(dim=0)[0]
            disjoint_aggs *= sign_to_mult
        else:
            raise ValueError(f"Merge method {merge_func} is not defined.")

        return disjoint_aggs
    
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    final_signs = resolve_sign(updated_checks)
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    
    return merged_tv

def top_5_percent_masked_centroid(cluster_points):
    n, d = cluster_points.shape
    top_5_percent = int(torch.ceil(torch.tensor(d * 0.05)).item())
    
    final_mask = torch.zeros((n, d), device=cluster_points.device)
    masked_rows = torch.zeros((n, d), device=cluster_points.device)

    for i in range(n):
        abs_values = torch.abs(cluster_points[i, :])
        top_indices = torch.argsort(abs_values, descending=True)[:top_5_percent]

        mask = torch.zeros(d, device=cluster_points.device)
        mask[top_indices] = 1

        final_mask[i, :] = mask
    
    normalized_mask = final_mask / n # take mean
    masked_rows = normalized_mask * cluster_points

    centroid = masked_rows.sum(dim=0)
    
    return centroid

def flatten_task_vectors(task_vectors):
    flat_tensors = []
    metadata = []

    for name in task_vectors[0].keys():
        stacked_values = torch.stack([vec[name].view(-1) for vec in task_vectors])
        flat_tensors.append(stacked_values)
        metadata.append((name, task_vectors[0][name].shape))

    flat_tensor = torch.cat(flat_tensors, dim=1)
    return flat_tensor, metadata

def recover_task_vector(centroid, metadata):
    recovered_dict = {}
    start = 0
    for name, shape in metadata:
        num_elements = torch.prod(torch.tensor(shape)).item()
        recovered_dict[name] = centroid[start:start + num_elements].view(shape)
        start += num_elements
    return recovered_dict



def create_basis(task_vectors, method):
    if method == "weight_averaging": # mean
        basis_vector = {name: torch.mean(torch.stack([vec[name] for vec in task_vectors]), dim=0)
                        for name in task_vectors[0].keys()}
    elif method == 'top5':
        flat_tensor, metadata = flatten_task_vectors(task_vectors)
        centroid = top_5_percent_masked_centroid(flat_tensor)
        basis_vector = recover_task_vector(centroid, metadata)

    elif method == 'ties':
        flat_tensor, metadata = flatten_task_vectors(task_vectors)
        centroid = ties_merging(flat_task_checks=flat_tensor,reset_thresh=20,merge_func='dis-mean')
        basis_vector = recover_task_vector(centroid, metadata)
        
    else:
        raise NotImplementedError(f"Merging method '{method}' is not implemented.")
    return basis_vector

def generate_random_cluster_assignments(task_names, k, num_assignments=3):
    random.seed(42)
    assignments = []
    for _ in range(num_assignments):
        if k is None:
            cluster_ids = range(1, len(task_names) + 1)
        else:
            cluster_ids = range(1, k + 1)
        random_assignment = {task: random.choice(cluster_ids) for task in task_names}
        assignments.append(random_assignment)
    return assignments


# Main basis merging function
def basis_merging(inner_method, num_split):
    k = num_split
    model_name = 'roberta-base'
    base_dir = f"/data-4/common/cindy2000_sh/tangent_task_arithmetic/{model_name}"

    csv_path = os.path.join(f"/data/common/cindy2000_sh/tangent_task_arithmetic/{model_name}", f"accuracy_comparison.csv")
    accuracy_df = pd.read_csv(csv_path)
    accuracy_df = accuracy_df[accuracy_df['fine_tune_accuracy'] >= accuracy_df['majority_label_accuracy']]

    accuracy_dataset_names = set(accuracy_df['dataset_name'])

    task_folders = [
        folder for folder in os.listdir(base_dir)
        if (os.path.isdir(os.path.join(base_dir, folder, str(args.seed)))) and (folder in accuracy_dataset_names)
    ]

    pretrained_model = get_pretrained_model()

    task_vectors = {}
    for task_name in task_folders:
        task_path = os.path.join(base_dir, task_name)
        checkpoint_path = os.path.join(task_path, 'best_model')
        if os.path.exists(checkpoint_path):
            task_vectors[task_name] = load_task_vector(checkpoint_path, pretrained_model)

    task_names = list(task_vectors.keys())

    cluster_assignments_path = os.path.join(base_dir, f"cluster_assignments_{k}.json")

    with open(cluster_assignments_path, "r") as f:
        ground_truth_assignment = json.load(f)

    if k == 1:
        random_assignments = [ground_truth_assignment]
    else:
        random_assignments = generate_random_cluster_assignments(task_names, k)
        random_assignments = [ground_truth_assignment] + random_assignments  

    unique_assignments = []
    seen_assignments = set()
    for assignment in random_assignments:
        assignment_tuple = tuple(sorted(assignment.items()))
        if assignment_tuple not in seen_assignments:
            seen_assignments.add(assignment_tuple)
            unique_assignments.append(assignment)

    for assignment_idx, cluster_assignments in enumerate(unique_assignments):
        assignment_type = "ground_truth" if assignment_idx == 0 else f"random_{assignment_idx - 1}"
        output_path = os.path.join(base_dir, f"{inner_method}_results_k{k}_{assignment_type}_{args.seed}.json")

        if os.path.exists(output_path):
            print(f"Skipping evaluation for {output_path} as it already exists.")
            continue

        clusters = {}
        for task, cluster_id in cluster_assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(task_vectors[task])

        cluster_bases = {cluster_id: create_basis(vectors, method=inner_method) for cluster_id, vectors in clusters.items()}

        task_vector_basis_merged = create_basis(list(cluster_bases.values()), method="weight_averaging")

        eval_results = {}

        dataset_performances = []
        for dataset_name in task_folders:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)

            global label_map # TODO: fix this later ...

            _, validation_dataset, label_map, tokenizer = load_and_tokenize_data(dataset_name, tokenizer)


            model = CustomRobertaForMaskedLM.from_pretrained(model_name, label_map=label_map)
            model.resize_token_embeddings(len(tokenizer))

            model = recover_model_from_task_vector(task_vector_basis_merged, model)

            training_args = CustomTrainingArguments(
                output_dir = f"/data-4/common/cindy2000_sh/tangent_task_arithmetic/{model_name}/{dataset_name}",
                report_to='none',
            )

            data_collator = CustomDataCollatorForPrompting(tokenizer=tokenizer, mlm=True)
            
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                eval_dataset = validation_dataset,
                data_collator=data_collator,
                compute_metrics = compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            )

            eval_output = trainer.evaluate()
            dataset_performance = eval_output.get("eval_accuracy", None)
            eval_results[dataset_name] = dataset_performance
            dataset_performances.append(dataset_performance)

        eval_results["average_performance"] = sum(dataset_performances) / len(dataset_performances)

        with open(output_path, "w") as f:
            json.dump(eval_results, f, indent=4)
        print(f"Results saved to {output_path}")



def pretrained_eval():
    model_name = 'roberta-base'
    base_dir = f"/data/common/cindy2000_sh/tangent_task_arithmetic/{model_name}"

    csv_path = os.path.join(base_dir, "accuracy_comparison.csv")
    accuracy_df = pd.read_csv(csv_path)
    accuracy_df = accuracy_df[accuracy_df['fine_tune_accuracy'] >= accuracy_df['majority_label_accuracy']]

    accuracy_dataset_names = set(accuracy_df['dataset_name'])

    task_folders = [
        folder for folder in os.listdir(base_dir)
        if (os.path.isdir(os.path.join(base_dir, folder))) and (folder in accuracy_dataset_names)
    ]

    eval_results = {}
    dataset_performances = []
    for dataset_name in task_folders:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        global label_map 
        _, validation_dataset, label_map, tokenizer = load_and_tokenize_data(dataset_name, tokenizer)


        model = CustomRobertaForMaskedLM.from_pretrained(model_name, label_map=label_map)
        model.resize_token_embeddings(len(tokenizer))


        training_args = CustomTrainingArguments(
            output_dir = f"/data/common/cindy2000_sh/tangent_task_arithmetic/{model_name}/{dataset_name}",
            report_to='none',
        )

        data_collator = CustomDataCollatorForPrompting(tokenizer=tokenizer, mlm=True)
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            eval_dataset = validation_dataset,
            data_collator=data_collator,
            compute_metrics = compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        eval_output = trainer.evaluate()
        dataset_performance = eval_output.get("eval_accuracy", None)
        eval_results[dataset_name] = dataset_performance
        dataset_performances.append(dataset_performance)

    eval_results["average_performance"] = sum(dataset_performances) / len(dataset_performances)

    output_path = os.path.join(base_dir, f"pretrained_metrics.json")

    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Results saved to {output_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Roberta on different datasets")
    
    # Add arguments for dataset, task, and training settings
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., 'anli')")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimizer")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--k", type=int, default=64, help="k-shot")
    parser.add_argument("--model_name", type=str, default='roberta-base')
    parser.add_argument("--main", type=str, required=True, choices=["train", "eval_majority", "basis_merging", "pretrained_merging"])
    parser.add_argument("--merge_method", type=str, choices=['weight_averaging','ties','top5'])
    parser.add_argument('--num_split',type=int, default=None)
    parser.add_argument('--seed',type=int)
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model_name = args.model_name
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    if args.dataset is not None:
        train_dataset, validation_dataset, label_map, _ = load_and_tokenize_data(args.dataset, tokenizer) # make sure label_map keys are sorted!!!
        
        if not is_valid_label_map(label_map):
            import pdb; pdb.set_trace()

    if args.main == 'train':
        train_pipeline()
    
    elif args.main == 'eval_majority':
        eval_majority_pipeline()
    
    elif args.main == 'basis_merging':
        basis_merging(args.merge_method, args.num_split)
    
    elif args.main == 'pretrained_merging':
        pretrained_eval()

  