# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import list_datasets, load_dataset, list_metrics, load_metric
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import math
import random
from tqdm import tqdm
import torch.nn.functional as F
transformers.logging.set_verbosity(transformers.logging.ERROR)
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
import os
#C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4
#os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7"

from src.dataset import load_data
from src.utils import bool_flag, get_output_file, load_checkpoints

    
def evaluate(model, tokenizer, testset, text_key=None, batch_size=10, pretrained=False, label_perm=(lambda x: x)):
    """
    Compute the accuracy of a model on a testset.  label_perm全称是label_permutation, 即标签置换、排列.
    """
    num_samples = len(testset)
    num_batches = int(math.ceil(num_samples / batch_size)) #math.ceil表示向上舍入到整数。eg:math.ceil(5.3)=6
    corr = []
    with torch.no_grad():
        for i in range(num_batches):
            lower = i * batch_size
            upper = min((i+1) * batch_size, num_samples)
            examples = testset[lower:upper]
            y = torch.LongTensor(examples['label'])
            if text_key is None:
                if pretrained:
                    x = tokenizer(examples['premise'], examples['hypothesis'], padding='max_length',
                                  max_length=256, truncation=True, return_tensors='pt')
                else:
                    x = tokenizer(examples['premise'], examples['hypothesis'], padding=True,
                                  truncation=True, return_tensors='pt')
            else:
                if pretrained:
                    x = tokenizer(examples[text_key], padding='max_length', max_length=256,
                                  truncation=True, return_tensors='pt')
                else:
                    x = tokenizer(examples[text_key], padding=True, truncation=True, return_tensors='pt')
            preds = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                          token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None)).logits.cpu()
            
            #distilbert前向传播函数没有token_type_ids
            # preds = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda()).logits.cpu()
            
            if label_perm is not None:
                corr.append(preds.argmax(1).eq(label_perm(y)))
            else:
                corr.append(preds.argmax(1).eq(y))

    return torch.cat(corr, 0)

class TokenDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['label'])

def evaluate_adv_samples(model, tokenizer, tokenizer_surr, adv_log_coeffs, clean_texts, labels,
                         attack_target=None, gumbel_samples=100, gumbel_batch_size=10, batch_size=10,
                         print_every=10, pretrained=False, pretrained_surrogate=False, label_perm=(lambda x: x)):

    #assert len(adv_log_coeffs) == len(labels) == len(clean_texts)
    num_samples = len(labels)
    all_corr = []
    if attack_target == '':
        all_sentences = []
    else:
        all_sentences = {'hypothesis': [], 'premise': []}
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            if attack_target == 'premise':
                log_coeffs = adv_log_coeffs['premise'][i].cuda().unsqueeze(0).repeat(gumbel_batch_size, 1, 1)
            elif attack_target == 'hypothesis':
                log_coeffs = adv_log_coeffs['hypothesis'][i].cuda().unsqueeze(0).repeat(gumbel_batch_size, 1, 1)
            else:
                log_coeffs = adv_log_coeffs[i].cuda().unsqueeze(0).repeat(gumbel_batch_size, 1, 1)
            adv_ids = []

            # Batch sampling of adv_ids
            num_batches = int(math.ceil(gumbel_samples / gumbel_batch_size))
            for j in range(num_batches):
                adv_ids.append(F.gumbel_softmax(log_coeffs, hard=True).argmax(-1))
            adv_ids = torch.cat(adv_ids, 0)[:gumbel_samples]
            evalset = {}
            
            text_key = None
            sentences = [tokenizer_surr.decode(adv_id) for adv_id in adv_ids]
            if attack_target == 'premise':
                premise = sentences
                hypothesis = [clean_texts['hypothesis'][i]] * gumbel_samples
                all_sentences['premise'].append(premise)
                all_sentences['hypothesis'].append(hypothesis)
                evalset['premise'] = premise
                evalset['hypothesis'] = hypothesis
            elif attack_target == 'hypothesis':
                premise = [clean_texts['premise'][i]] * gumbel_samples
                hypothesis = sentences
                all_sentences['premise'].append(premise)
                all_sentences['hypothesis'].append(hypothesis)
                evalset['premise'] = premise
                evalset['hypothesis'] = hypothesis
            else:
                all_sentences.append(sentences)
                evalset['text'] = sentences
                text_key = 'text'
            evalset['label'] = [labels[i]] * gumbel_samples
            evalset = TokenDataset(evalset)
            corr = evaluate(model, tokenizer, evalset, text_key, batch_size,
                            pretrained=pretrained, label_perm=label_perm)
            all_corr.append(corr.unsqueeze(0))
            if (i+1) % print_every == 0:
                Adv_acc = torch.cat(all_corr, 0).float().mean(1).eq(1).float().mean()
                print('Adversarial accuracy = %.4f' % torch.cat(all_corr, 0).float().mean(1).eq(1).float().mean())
    
    all_corr = torch.cat(all_corr, 0)
    _, min_index = all_corr.float().cummin(1)
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embed = hub.load("utility/universal-sentence-encoder_4")
    if attack_target == 'premise':
        with tf.device('/cpu:0'):
            clean_embeddings = embed(clean_texts['premise'])
        adv_texts = [all_sentences['premise'][j][min_index[j, -1]] for j in range(num_samples)]
    elif attack_target == 'hypothesis':
        with tf.device('/cpu:0'):
            clean_embeddings = embed(clean_texts['hypothesis'])
        adv_texts = [all_sentences['hypothesis'][j][min_index[j, -1]] for j in range(num_samples)]
    else:
        with tf.device('/cpu:0'):
            clean_embeddings = embed(clean_texts)
        adv_texts = [all_sentences[j][min_index[j, -1]] for j in range(num_samples)]
    with tf.device('/cpu:0'):
        adv_embeddings = embed(adv_texts)
    cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))
    print('Cosine similarity = %.4f' % cosine_sim)  

    #用来筛选不合格的样本，提高相似度
    result_list = tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1)
    result = tf.reduce_sum(result_list)
    count = 0
    count_sub =0
    query =0
    for i in range(len(result_list)):
        if result_list[i] < 0.20:          #设置过滤阈值
            query += 100
            result = result - result_list[i]
            if False in all_corr[i]:
                count_sub += 1
            count += 1
        else:
            for j in range(len(all_corr[i])):
                if all_corr[i][j] == False:
                    query = query + j + 1
                    break
    cosine_sim = result / (len(result_list) - count)
    Adv_acc = Adv_acc + count_sub / len(result_list)
    query_avg = query/len(result_list)
    print('过滤之后的 Adv_acc = %.4f' % Adv_acc)  
    print("平均查询次数= %.4f" % query_avg)

    print('过滤后的Cosine similarity = %.4f' % cosine_sim)      
    return all_sentences, all_corr, cosine_sim


def main(args):
    # Load data
    dataset, num_labels = load_data(args)
    if args.dataset == 'mnli':
        text_key = None
        testset_key = 'validation_%s' % args.mnli_option

    elif args.dataset == "FakeNews":    #新增数据集
        text_key = "text"
        testset_key = 'validation'

    else:
        text_key = 'text' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'sentence'
        testset_key = 'test' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'validation'
    
    # Load target model
    pretrained = args.target_model.startswith('textattack')
    pretrained_surrogate = args.surrogate_model.startswith('textattack')
    suffix = '_finetune' if args.finetune else ''
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.target_model, num_labels=num_labels).cuda()
    if not pretrained:
        model_checkpoint = os.path.join(args.result_folder, '%s_%s%s.pth' % (args.target_model.replace('/', '-'), args.dataset, suffix))
        print('Loading checkpoint: %s' % model_checkpoint)
        model.load_state_dict(torch.load(model_checkpoint))
        tokenizer.model_max_length = 512
    if args.target_model == 'gpt2':
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    label_perm = lambda x: x
    if pretrained:
        #ori
        # if args.target_model == 'textattack/bert-base-uncased-MNLI' or args.target_model == 'textattack/xlnet-base-cased-MNLI':
        #     label_perm = lambda x: (x + 1) % 3
        # elif args.target_model == 'textattack/roberta-base-MNLI':
        #     label_perm = lambda x: -(x - 1) + 1
        #if mnli数据集
        if args.target_model == 'bert-base-uncased' or args.target_model == 'textattack/xlnet-base-cased-MNLI':
            label_perm = lambda x: (x + 1) % 3
        elif args.target_model == 'roberta-base':
            label_perm = lambda x: -(x - 1) + 1


    # Compute clean accuracy
    corr = evaluate(model, tokenizer, dataset[testset_key], text_key, pretrained=pretrained, label_perm=label_perm)
    print('Clean accuracy = %.4f' % corr.float().mean())
    
    surr_tokenizer = AutoTokenizer.from_pretrained(args.surrogate_model, use_fast=True)
    surr_tokenizer.model_max_length = 512
    if args.surrogate_model == 'gpt2':
        surr_tokenizer.padding_side = "right"
        surr_tokenizer.pad_token = tokenizer.eos_token
        
    clean_texts, adv_texts, clean_logits, adv_logits, adv_log_coeffs, labels, times = load_checkpoints(args)

    label_perm = lambda x: x
    if pretrained and args.surrogate_model != args.target_model:
        if args.target_model == 'textattack/bert-base-uncased-MNLI' or args.target_model == 'textattack/xlnet-base-cased-MNLI':
            label_perm = lambda x: (x + 1) % 3
        elif args.target_model == 'textattack/roberta-base-MNLI':
            label_perm = lambda x: -(x - 1) + 1
    
    attack_target = args.attack_target if args.dataset == 'mnli' else ''
    all_sentences, all_corr, cosine_sim = evaluate_adv_samples(
        model, tokenizer, surr_tokenizer, adv_log_coeffs, clean_texts, labels, attack_target=attack_target,
        gumbel_samples=args.gumbel_samples, batch_size=args.batch_size, print_every=args.print_every,
        pretrained=pretrained, pretrained_surrogate=pretrained_surrogate, label_perm=label_perm)
    
    print("__logs:" + json.dumps({
        "cosine_similarity": float(cosine_sim),
        "adv_acc2": all_corr.float().mean(1).eq(1).float().mean().item()
    }))
    output_file = get_output_file(args, args.surrogate_model, args.start_index, args.end_index)
    output_file = os.path.join(args.adv_samples_folder,
                               'transfer_%s_%s' % (args.target_model.replace('/', '-'), output_file))
    torch.save({
        'all_sentences': all_sentences, 
        'all_corr': all_corr, 
        'clean_texts': clean_texts,        
        'labels': labels, 
        'times': times
    }, output_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate white-box attack.")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder for loading trained models")
    parser.add_argument("--adv_samples_folder", default="adv_samples/", type=str,
        help="folder for saving generated samples")
    parser.add_argument("--dump_path", default="log/", type=str,
        help="Path to dump logs")


    # Data 
    parser.add_argument("--data_folder",  type=str, default= "data/",#required=True,
        help="folder in which to store data")
    parser.add_argument("--dataset", default="yelp", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "sst2", "FakeNews"],
        help="classification dataset to use")
    parser.add_argument("--mnli_option", default="mismatched", type=str,
        choices=["matched", "mismatched"],
        help="use matched or mismatched test set for MNLI")

    # Model
    parser.add_argument("--target_model", default="gpt2", type=str,
        choices=["gpt2","xlm-clm-ende-1024","textattack/bert-base-uncased-ag-news", "textattack/bert-base-uncased-yelp-polarity",
        "textattack/bert-base-uncased-MNLI", "bert-base-uncased", "roberta-base", 'albert-base-v2', 'distilbert-base-uncased'],
        help="type of model")
    parser.add_argument("--surrogate_model", default="gpt2", type=str,
        help="type of model")
    parser.add_argument("--finetune", default=True, type=bool_flag,
        help="load finetuned model")

    # Attack setting
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--end_index", default=80, type=int,
        help="end sample index")
    parser.add_argument("--num_samples", default=80, type=int,
        help="number of samples per split")
    parser.add_argument("--num_iters", default=100, type=int,
        help="number of epochs to train for")
    parser.add_argument("--batch_size", default=10, type=int,#original 10
        help="batch size for evaluation")
    parser.add_argument("--attack_target", default="premise", type=str,
        choices=["premise", "hypothesis"],
        help="attack either the premise or hypothesis for MNLI")
    parser.add_argument("--adv_loss", default="cw", type=str,
        choices=["cw", "ce"],
        help="adversarial loss")
    parser.add_argument("--constraint", default="cosine", type=str,
        choices=["cosine", "bertscore", "bertscore_idf"],
        help="constraint function")
    parser.add_argument("--lr", default=3e-1, type=float,
        help="learning rate")
    parser.add_argument("--kappa", default=5, type=float,
        help="CW loss margin")
    parser.add_argument("--embed_layer", default=-1, type=int,
        help="which layer of LM to extract embeddings from")
    parser.add_argument("--lam_sim", default=1, type=float,#origin=1
        help="embedding similarity regularizer")
    parser.add_argument("--lam_perp", default=1, type=float,#origin=1
        help="(log) perplexity regularizer")
    parser.add_argument("--print_every", default=80, type=int,
        help="print result every x samples")
    parser.add_argument("--gumbel_samples", default=100, type=int,
        help="number of gumbel samples; if 0, use argmax")

    args = parser.parse_args()

    main(args)