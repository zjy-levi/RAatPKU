# Description: 使用 Hugging Face Transformers 库微调 BERT 模型
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bert_finetune.py
@Time    :   2024/09/17 19:11:07
@Author  :   _zJy_
@Version :   1.0
@License :   (C)Copyright 2024, MIT License
@Contact :   jinyuzh@hku.hk
@Desc    :   使用huggingface的transformers库微调BERT模型，自定义了Trainer类的compute_loss方法。自定义了PyTorch模型类，继承了BertPreTrainedModel类，实现了forward方法。使用BCEWithLogitsLoss作为损失函数，使用accuracy作为评估指标。微调后的模型保存在model/fine_tuned_bert文件夹中。唯一的遗憾是没能跑通peft微调，不知道是不是因为版本问题。值得注意的是多卡训练一定要指定CUDA_VISIBLE_DEVICES环境变量，否则会报错。
'''

import os
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
# 限制使用的GPU,为了防止cuda 报错啊啊啊！！！
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# 为config添加额外的参数
def add_config(config, extra_params):
    """向config中添加额外的参数

    Args:
        config (AutoModelConfig): 预训练模型的配置
        extra_params (dict): 需要额外传入的配置参数

    Returns:
        AutoModelConfig: 传入额外参数后的配置
    """
    for key, value in extra_params.items():
        if hasattr(config, key):
            print(f"Warning: Key '{key}' already exists in the config with value '{getattr(config, key)}'. It will be overwritten.")
        setattr(config, key, value)
    return config
# Convert labels to multi-label format
def convert_labels_to_multilabel(examples):
    labels = examples['labels']
    multi_labels = np.zeros((len(labels), 28))  # Assuming 28 emotion labels
    for i, label_list in enumerate(labels):
        for label in label_list:
            multi_labels[i][label] = 1
    # 使用tokenizer对文本进行编码
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128, return_tensors="pt")
    # 更新examples字典
    examples['input_ids'] = tokenized_inputs['input_ids'].tolist()
    examples['attention_mask'] = tokenized_inputs['attention_mask'].tolist()
    if 'token_type_ids' in tokenized_inputs:
        examples['token_type_ids'] = tokenized_inputs['token_type_ids'].tolist()
    examples['labels'] = multi_labels.tolist()
    return examples

# 定义自定义的PyTorch模型类
class CustomBERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBERTModel, self).__init__(config)
        self.config = config
        print(config)
        self.bert = BertModel.from_pretrained(config.model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.cls_num_labels)
        self.init_weights()
        # self.config = self.bert.config
        # self.config.architectures = ["CustomBERTModel"]
        # self.config['num_labels'] = num_labels
        print(self.config)
    
    def forward(self, **inputs):
        if "labels" in inputs.keys():
            inputs.pop("labels")
        outputs = self.bert(**inputs)
        pooled_output = outputs[1]  # 获取池化的输出
        logits = self.classifier(pooled_output)
        return {"logits":logits}
    
if __name__ == "__main__":
    # 下载 BERT 预训练模型和分词器
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    extra_config = {
        "cls_num_labels": 28,
        "model_name": model_name
    }
    config = add_config(config, extra_config)
    model = CustomBERTModel(config=config)
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    dataset = dataset.map(convert_labels_to_multilabel, batched=True)
    traindataset = dataset['train']
    testdataset = dataset['test']
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1, 
        target_modules=["classifier"]
    )
    # 应用 LoRA 到模型
    model = get_peft_model(model, lora_config)
    # 查看LoRA应用前后模型的可训练参数量
    print("Number of trainable parameters before LoRA:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(model)
    # Define custom Trainer to use BCEWithLogitsLoss
    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return (loss, outputs) if return_outputs else loss
    # # Setup evaluation 
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = (logits > 0).astype(int)
        return metric.compute(predictions=predictions, references=labels)

    # 微调模型
    training_args = TrainingArguments(
        output_dir='./results',
        save_steps=500, # 每500步保存一次模型到/results文件夹,输出checkpoint_{steps}文件夹
        save_total_limit=2, # 保存最近的2个模型
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        logging_steps=100,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=traindataset,
        eval_dataset=testdataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 保存微调后的模型
    model.save_pretrained('model/fine_tuned_bert') # 只有model继承PretrainedModel才能使用这个方法
    tokenizer.save_pretrained('model/fine_tuned_bert')

    # # # 使用微调后的模型进行预测
    # # texts = [
    # #     "I love programming!",
    # #     "I feel sad today."
    # # ]

    # # inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    # # outputs = model(**inputs)
    # # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # # # 打印预测结果
    # # for i, text in enumerate(texts):
    # #     print(f"Text: {text}")
    # #     print(f"Predicted probabilities: {predictions[i].detach().numpy()}")