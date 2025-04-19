"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
# from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
# from transformers.models.outputs.modeling_outputs import SequenceClassifierOutput
from transformers import GPT2LMHeadModel

import logging
logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        
        self.classifier = None
        #torch.nn.Linear(config.hidden_size, config.num_labels)
        #RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None
        self.return_representation = None
        
        self.initial_parameters_copy = []#[ p.detach().clone() for p in self.roberta.parameters() ]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        # print(outputs.shape)
        # print (sequence_output.shape)
        #print (mask_pos)
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        # print (sequence_mask_output.shape)
        sequence_CLS_output  = sequence_output[torch.arange(sequence_output.size(0)), 0]

        # if self.model_args.use_lm_head:
        # Logits over vocabulary tokens
        if self.return_representation:
            return sequence_mask_output
        
        
        prediction_mask_scores = self.lm_head(sequence_mask_output)
        # print (prediction_mask_scores.shape)
        
        
        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        # elif self.model_args.use_CLS_linearhead:
        #     logits = self.classifier(sequence_CLS_output)
       
            
        
        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # if self.model_args.l1_reg != 0.0:
            #     l1_norm = sum(torch.sum(torch.abs(p - q.to('cuda'))) for p, q in zip( self.roberta.parameters(), self.initial_parameters_copy ) )
            #     loss += self.model_args.l1_reg * l1_norm
            
        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        
        
        return ((loss,) + output) if loss is not None else output


class MultiHeadRobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config, num_heads):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        
        self.classifier = None
        #torch.nn.Linear(config.hidden_size, config.num_labels)
        #RobertaClassificationHead(config)
        self.lm_heads = [RobertaLMHead(config)] * num_heads
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_lists = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None
        self.return_representation = None
        
        self.initial_parameters_copy = []#[ p.detach().clone() for p in self.roberta.parameters() ]

    def forward(
        self,
        head_idx,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        # print(outputs.shape)
        # print (sequence_output.shape)
        #print (mask_pos)
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        # print (sequence_mask_output.shape)
        sequence_CLS_output  = sequence_output[torch.arange(sequence_output.size(0)), 0]

        # if self.model_args.use_lm_head:
        # Logits over vocabulary tokens
        if self.return_representation:
            return sequence_mask_output
        
        
        prediction_mask_scores = self.lm_heads[head_idx](sequence_mask_output)
        # print (prediction_mask_scores.shape)
        
        
        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_lists[head_idx])):
            logits.append(prediction_mask_scores[:, self.label_word_lists[head_idx][label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        # elif self.model_args.use_CLS_linearhead:
        #     logits = self.classifier(sequence_CLS_output)
       
            
        
        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # if self.model_args.l1_reg != 0.0:
            #     l1_norm = sum(torch.sum(torch.abs(p - q.to('cuda'))) for p, q in zip( self.roberta.parameters(), self.initial_parameters_copy ) )
            #     loss += self.model_args.l1_reg * l1_norm
            
        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        
        
        return ((loss,) + output) if loss is not None else output
