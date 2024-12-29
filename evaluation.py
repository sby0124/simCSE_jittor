import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import jittor as jt
from transformers import AutoModel, AutoTokenizer,AutoConfig
from simcse.models import BertForCL
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG, handlers=[logging.StreamHandler(sys.stdout)])

from train import ModelArguments
# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=[ 'test', 'fasttest'],
            default='test', )
    parser.add_argument("--task_set", type=str, 
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
                        default=["STSBenchmark"],
            # default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
            #          'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
            #          'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    parser.add_argument("--file_path_of_model", type=str, 
            help="File path of the model to evaluate")
    args = parser.parse_args()
    
    # Load transformers' model checkpoint
    # model = AutoModel.from_pretrained(args.model_name_or_path)#加载预训练模型
    argvs ={"do_mlm":False,"pooler_type":"cls"}
    model_args = ModelArguments(argvs)
    config=AutoConfig.from_pretrained("unsup-simcse-bert-base-uncased")
    model = BertForCL(config, model_args)
    model.load_state_dict(jt.load("/home/aiuser/SimCSE/model_pre_2_46000.pth"))
    tokenizer = AutoTokenizer.from_pretrained("unsup-simcse-bert-base-uncased")#加载预训练模型的tokenizer
    # 
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    # Set up the tasks
    args.tasks = ["STSBenchmark","STS16",'STS12', 'STS13', 'STS14', 'STS15','SICKRelatedness']

    # Set params for SentEval
        # Full mode
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                        'tenacity': 5, 'epoch_size': 4}

    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:#如果max_length不为空，将句子转换为token
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )#将句子转换为token，
        else:#如果max_length为空，将句子转换为token
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
        
        # Move to the correct device
        # for k in batch:
        #     batch[k] = batch[k]#.to(device)
        input_ids = jt.array(batch['input_ids'].numpy())
        attention_mask = jt.array(batch['attention_mask'].numpy())
        token_type_ids = jt.array(batch['token_type_ids'].numpy())
        batch['input_ids'] = input_ids
        batch['attention_mask'] = attention_mask
        batch['token_type_ids'] = token_type_ids
        if jt.has_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            model.cuda()
        # Get raw embeddings，获取句子的embedding
        with jt.no_grad():
            outputs = model.execute(input_ids, attention_mask, token_type_ids,sent_emb=True)
            last_hidden = outputs["last_hidden_state"]
            pooler_output = outputs["pooler_output"]
            hidden_states = outputs["hidden_states"]
        return pooler_output

    results = {}

    for task in args.tasks:
        print("------ %s ------" % (task))
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
   
    if args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
