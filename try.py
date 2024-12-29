from bert_model.bert_model import BertForQuestionAnswering, BertConfig
from jittor.optim import AdamW
import jittor as jt
from jittor import nn
loss_fn = nn.CrossEntropyLoss()
pre=[[[0.1],[0.2],[0.7]],[[0.7],[0.2],[0.1]],[[0.2],[0.7],[0.1]]]
label=[2,0,1]
pre=jt.array(pre)
label=jt.array(label)
print(loss_fn(pre,label))