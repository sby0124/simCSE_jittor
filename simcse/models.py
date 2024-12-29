import jittor as jt
import jittor.nn as nn
import numpy as np

from bert_model.bert_model import BertModel,BertBase,BertLMPredictionHead
#引入斯皮尔曼相关系数
from scipy.stats import spearmanr
class MLPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def execute(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = float(temp)
    def execute(self, x, y):
        # 手动实现 Cosine Similarity
        x_norm = jt.norm(x, p=2, dim=-1, keepdim=True)  # x 的模长
        y_norm = jt.norm(y, p=2, dim=-1, keepdim=True)  # y 的模长
        dot_product = jt.sum(x * y, dim=-1, keepdim=True)  # 点积
        cosine_sim = dot_product / (x_norm * y_norm + 1e-8)  # 余弦相似度
        return cosine_sim / self.temp
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def execute(self, attention_mask, outputs):
        last_hidden = outputs["last_hidden_state"]
        hidden_states = outputs["hidden_states"]

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
import numpy as np

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)
    mlm_outputs = None
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    z1, z2 = pooler_output[:,0], pooler_output[:,1]
    
    if num_sent == 3:
        z3 = pooler_output[:, 2]
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    cos_sim = cos_sim.view(cos_sim.size(0), -1)
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        z1_z3_cos = z1_z3_cos.view(z1_z3_cos.size(0), -1)
        cos_sim = jt.concat([cos_sim, z1_z3_cos], 1)
    labels = jt.arange(cos_sim.size(0)).long()
    loss_fct = nn.CrossEntropyLoss()
    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = jt.var(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights
    loss = loss_fct(cos_sim, labels)
    #将cos_sim写道一个txt文件中
    label_array = labels.numpy()
    pre_label= np.argmax(cos_sim, axis=1)
    #获取分类正确的数目
    correct = (pre_label == label_array).sum()
    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs["last_hidden_state"])
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        mlm_weights = float(cls.model_args.mlm_weight)
        loss = loss + mlm_weights * masked_lm_loss
    
    #返回一个字典
    return{
        "loss": loss,
        "logits": cos_sim,
        "hidden_states": outputs["hidden_states"],
        "pooler_output": pooler_output,
        "last_hidden_state": outputs["last_hidden_state"],
        "correct": correct,
    }

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
):
    
    
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    return {
        "pooler_output": pooler_output,
        "hidden_states": outputs["hidden_states"],
        "last_hidden_state": outputs["last_hidden_state"],
    }


class BertForCL(BertBase):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, model_args):
        super().__init__(config)
        self.model_args = model_args
        self.bert = BertModel(config, add_pooling_layer=False)
        if model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        cl_init(self, config)

    def execute(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mlm_input_ids=None,
        mlm_labels=None,
        sent_emb=False,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
