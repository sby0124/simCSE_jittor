import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
# import jt
import jittor as jt
# import collections
# import random
# import torch
from datasets import load_dataset
import inspect
#判断是否有gpu
from jittor.dataset import VarDataset


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel,
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from simcse.models import  BertForCL#,RobertaForCL
import tqdm
import jittor.optim as optim
from jittor.dataset import Dataset
jt.gc()
logger = logging.getLogger(__name__)#
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())#
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
class CustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.set_attrs(total_len=len(data))  # 设置数据集的总长度

    def __getitem__(self, idx):
        # 返回数据的逻辑
        return self.data[idx]
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    def __init__(self, args):
        #对于args里的参数，如果自己有这个成员变量就赋值，没有就赋默认值
        #遍历自己的成员变量
        for name in self.__dataclass_fields__:
            #如果args里有这个成员变量
            if name in args:
                #赋值
                #需要考虑一下，如果是数字类型的参数，需要转换成数字
                if self.__dataclass_fields__[name].default == int:
                    setattr(self, name, int(args[name]))
                elif self.__dataclass_fields__[name].default == float:
                    setattr(self, name, float(args[name]))
                else:
                    setattr(self, name, args[name])
            else:
                #赋默认值
                pass
        
    model_name_or_path: Optional[str] = field(#模型的名字或路径
        default="unsup-simcse-bert-base-uncased",
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )


@dataclass
class DataTrainingArguments:#这是一个数据类，用来存储数据集参数
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    def __init__(self, args):
        #对于args里的参数，如果自己有这个成员变量就赋值，没有就赋默认值
        #遍历自己的成员变量
        for name in self.__dataclass_fields__:
            #如果args里有这个成员变量
            if name in args:
                #赋值
                #需要考虑一下，如果是数字类型的参数，需要转换成数字
                if type(self.__dataclass_fields__[name].default) == int:
                    setattr(self, name, int(args[name]))
                elif type(self.__dataclass_fields__[name].default) == float:
                    setattr(self, name, float(args[name]))
                else:
                    setattr(self, name, args[name])
            else:
                #赋默认值
                pass
        pass
    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments:#这是一个数据类，用来存储训练参数
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    def __init__(self, args):
        self.metric_for_best_model = args["metric_for_best_model"]
        
        #对于args里的参数，如果自己有这个成员变量就赋值，没有就赋默认值
        #遍历自己的成员变量
        for name in self.__dataclass_fields__:
            #如果args里有这个成员变量
            if name in args:
                #赋值
                #需要考虑一下，如果是数字类型的参数，需要转换成数字
                if type(self.__dataclass_fields__[name].default) == int:
                    setattr(self, name, int(args[name]))
                elif type(self.__dataclass_fields__[name].default) == float:
                    setattr(self, name, float(args[name]))
                else:
                    setattr(self, name, args[name])
            else:
                #赋默认值
                setattr(self, name, self.__dataclass_fields__[name].default)
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    def __post_init__(self):
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    num_train_epochs:int = field(
        default=1,
    )
    
    do_train: Optional[bool] = field(
        default=True, metadata={"help": "train or not."}
    )
    overwrite_output_dir:Optional[bool] = field(
        default=True, metadata={"help": "train or not."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    # num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
        },
    )
    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )

def parse_args(argv):
    args = {}
    # 遍历命令行参数
    for i in range(1, len(argv)):  # 从1开始，因为argv[0]是脚本名称
        if argv[i].startswith('--'):
            # 如果参数后面有值，则获取该值
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                #去除--，并存入字典
                key = argv[i].replace('--', '')
                args[key] = argv[i + 1]
            else:
                pass
    return args


def main():
    parser = parse_args(sys.argv)
    model_args = ModelArguments(parser)
    data_args = DataTrainingArguments(parser)
    training_args = OurTrainingArguments(parser)
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    set_seed(1111)#设置随机种子
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")
    config=AutoConfig.from_pretrained("unsup-simcse-bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("unsup-simcse-bert-base-uncased")#加载预训练模型的tokenizer
    model = BertForCL(config,model_args)#
    dict = jt.load("unsup_bert_model.pth")
    i = 0
    j = 0
    for key in model.state_dict():
        j += 1  
        pre_key = key.replace("bert.","")
        pre_key = pre_key.replace("mlp","pooler")
        if pre_key in dict.keys():
            model.state_dict()[key] = dict[pre_key]
            i += 1  
        if i!=j:
            print(key)
            print(pre_key)
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        #这是将两个相同的句子拼接在一起
        sentences = examples[sent0_cname] + examples[sent1_cname]
        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]#将两个相同的句子分在一组
        return features
    if training_args.do_train:
        print("map begin")
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,#多进程处理        
            remove_columns=column_names,#删除column_names
            load_from_cache_file= True,#是否从缓存文件中加载
        )
        print("map end")
    # train_dataset = train_dataset.select(range(10000))#选择的是前10个数据
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], jt.array]]]) -> Dict[str,jt.array]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in tqdm.tqdm(features, desc="Preparing features"):
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])
            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch     
        def mask_tokens(
            self, inputs: jt.array, special_tokens_mask: Optional[jt.array] = None
        ) -> Tuple[jt.array, jt.array]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            inputs = jt.array(inputs.numpy())
            labels = jt.array(labels.numpy())
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            shape = labels.shape
            #将torch.size转换为jittor.size
            shape = [shape[i] for i in range(len(shape))]
            probability_matrix = jt.full(shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = jt.array(special_tokens_mask, dtype=jt.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = jt.bernoulli(probability_matrix).astype(jt.bool)
            labels[jt.logical_not(masked_indices)] = -100  # We only compute loss on masked tokens
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = jt.bernoulli(jt.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            # 10% of the time, we replace masked input tokens with random word
            indices_random = jt.bernoulli(jt.full(shape, 0.5)).bool() & masked_indices & jt.logical_not(indices_replaced)
            random_words = jt.randint(0,len(self.tokenizer), shape, dtype=jt.int64)
            inputs[indices_random] = random_words[indices_random]
            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels
    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)
    #将train_dataset["input_ids"]转换为numpy数组
    train_dataset = data_collator(train_dataset)
    batch_size = 64
    input_ids_data = CustomDataset(jt.array(train_dataset["input_ids"].numpy())).set_attrs(batch_size=batch_size, shuffle=False)
    attention_mask_data = CustomDataset(jt.array(train_dataset["attention_mask"].numpy())).set_attrs(batch_size=batch_size, shuffle=False)
    token_type_ids_data = CustomDataset(jt.array(train_dataset["token_type_ids"].numpy())).set_attrs(batch_size=batch_size, shuffle=False)
    flag = False
    if("mlm_input_ids" in train_dataset.keys()):
        flag = True
        mlm_input_ids_data = CustomDataset(jt.array(train_dataset["mlm_input_ids"].numpy())).set_attrs(batch_size=batch_size, shuffle=False)
        mlm_labels_data = CustomDataset(jt.array(train_dataset["mlm_labels"].numpy())).set_attrs(batch_size=batch_size, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=3e-6)
    if jt.has_cuda:
        model.cuda()
    j = 0
    for i in range(3):
        if flag:
            for input_ids,attention_mask,token_type_ids,mlm_input_ids,mlm_labels in tqdm.tqdm(zip(input_ids_data,attention_mask_data,token_type_ids_data,mlm_input_ids_data,mlm_labels_data)):
                optimizer.zero_grad()
                outputs = model.execute(input_ids,attention_mask,token_type_ids,mlm_input_ids,mlm_labels)
                loss = outputs["loss"]
                optimizer.step(loss)
                j+=1
                if j % 3000 == 0:
                    jt.save(model.state_dict(), f"model_pre_mlp_{i}_{j}.pth")
        else:
            #使用tqdm显示进度条
            for input_ids, attention_mask, token_type_ids in tqdm.tqdm(zip(input_ids_data, attention_mask_data, token_type_ids_data), total=len(input_ids_data)):
                optimizer.zero_grad()
                outputs = model.execute(input_ids,attention_mask,token_type_ids)
                loss = outputs["loss"]
                optimizer.step(loss)
                j+=1
                if j % 3000 == 0:
                #每64000个句子保存一次模型
                    jt.save(model.state_dict(), f"model_pre_mlp_{i}_{j}.pth")
        
if __name__ == "__main__":
    main()
