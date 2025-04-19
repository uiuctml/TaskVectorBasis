import torch
from args import parse_arguments, parse_data_arguments
from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from transformers import set_seed
from src.models import RobertaForPromptFinetuning
import time
from pathlib import Path
from src.dataset import FewShotDataset
from transformers import Trainer
from src.processors import compute_metrics_mapping
from typing import Callable, Dict

import sys
import os
sys.path.append('/home/cindy2000_sh/TaskVectorBasis/L&S')
from localize_utils import *

set_seed(0)

root = "/data/common/lm-bff"
modelname = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(modelname)

def initialize_model(model_path):
    config = AutoConfig.from_pretrained(
            modelname,
            attn_implementation="eager"
        )
    
    model = RobertaForPromptFinetuning.from_pretrained(
        model_path,
        config=config,
    )
    
    return model

def select_trainable_parameters(model):
    params = {}
    for n, p in model.named_parameters():
        if 'encoder.layer' in n:
            params[n] = p
                    
    return params

train_mask = True
graft_args = parse_arguments()
graft_args.sigmoid_bias = 3
graft_args.learning_rate = 1e7
graft_args.l1_strength = 0
graft_args.num_train_epochs = 10
graft_args.sparsity = 1e-5

args = parse_arguments()
args.batch_size = 8
args.model_name = modelname
args.save_mask = True
args.save_model = False

# TODO: change to your checkpoint folders
ckpt_pth = "/data/common/lm-bff/ckpt_paths/log_noembed_SGD_graft/"
mask_folder = f"~/Localize-and-Stitch/1e-2_new/"
task_list = ["SST-2", "cr", "mr", "mpqa", "trec", "subj", "QNLI", "SNLI", "MNLI", "RTE", "MRPC", "QQP"]

final_model = initialize_model(modelname)
pretrained_model = initialize_model(modelname)
finetuned_models = [initialize_model(ckpt_pth+f"{dataset_name}-prompt-64-0-{modelname}-2-2e-5") for dataset_name in task_list]
trainable_params = select_trainable_parameters(pretrained_model)

start_time = time.time()
masks = []
for i in range(len(task_list)):
    dataset_name = task_list[i]
    finetuned_model = finetuned_models[i]

    # To optimize for a mask
    print(f"------------Localizing for {dataset_name}------------")
    if train_mask:
        data_args = parse_data_arguments(dataset_name)
        valset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=False)
        )
        final_model.label_word_list = torch.tensor(valset.label_word_list).long().cuda()
        trainer = Trainer(model=pretrained_model, train_dataset=valset)
        val_dataloader = trainer.get_train_dataloader()

        localizer = Localizer(trainable_params, final_model, pretrained_model, finetuned_model, dataset_name, args, graft_args, model_type='roberta')
        mask, proportion, test = localizer.train_graft(val_dataloader, dataset_name)
        if args.save_mask:
            Path(mask_folder).mkdir(parents=True, exist_ok=True)    
            torch.save(mask, mask_folder+f'mask_{dataset_name.lower()}-prompt-64-0-roberta-base-2-2e-5')

    # To just load trained masks
    else:
        mask = torch.load(mask_folder+f'mask_{dataset_name.lower()}-prompt-64-0-roberta-base-2-2e-5')
    
    masks.append(mask)

localize_time = time.time() - start_time
# TODO: change your cluster result
combined_idx = [(0,1,2,3),(4,5),(6,10,11),(7,8,9)]
stitcher = MultiHeadStitcher(trainable_params, final_model, pretrained_model, finetuned_models, masks, combined_idx)
merged_model = stitcher.interpolate_models() # this is the combined model
stitch_time = time.time() - start_time - localize_time

if args.save_model:
    # TODO: change to your checkpoint folders
    path = f"/your/checkpoint/folder/"
    merged_model.save_pretrained(path, safe_serialization=False)
    tokenizer.save_pretrained(path)

def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        # Note: the eval dataloader is sequential, so the examples are in order.
        # We average the logits over each sample for using demonstrations.
        predictions = p.predictions
        num_logits = predictions.shape[-1]
        logits = predictions.reshape([test_dataset.num_sample, -1, num_logits])
        logits = logits.mean(axis=0)
        
        if num_logits == 1:
            preds = np.squeeze(logits)
        else:
            preds = np.argmax(logits, axis=1)

        # Just for sanity, assert label ids are the same.
        label_ids = p.label_ids.reshape([test_dataset.num_sample, -1])
        label_ids_avg = label_ids.mean(axis=0)
        label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
        assert (label_ids_avg - label_ids[0]).mean() < 1e-2
        label_ids = label_ids[0]
        
        return compute_metrics_mapping[task_name](task_name, preds, label_ids)

    return compute_metrics_fn

acc_list = []
for dataset_name in task_list:
    print(f"------------Evaluating on {dataset_name}------------")
    data_args = parse_data_arguments(dataset_name)
    test_dataset = (
        FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=False)
    )
    merged_model.label_word_list = torch.tensor(test_dataset.label_word_list).long().cuda()

    trainer = Trainer(model=merged_model, eval_dataset=test_dataset, compute_metrics=build_compute_metrics_fn(dataset_name.lower()))
    output = trainer.evaluate(eval_dataset=test_dataset)

    if dataset_name == "MNLI":
        acc_list.append(output['eval_mnli/acc'])
    elif dataset_name == "MRPC":
        acc_list.append(output['eval_f1'])
    else:
        acc_list.append(output['eval_acc'])

    print(output)

print("Average performance is ", sum(acc_list)/len(acc_list))
