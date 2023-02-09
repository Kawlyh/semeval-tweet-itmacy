import os
import sys
import logging
import datasets
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding, \
    EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 读取训练和测试数据集
#train = pd.read_csv('corpus/triple-data-noise-20.csv')
train = pd.read_csv("corpus/new_triple_data.csv")
val = pd.read_csv("corpus/new_val.csv")
test = pd.read_csv("corpus/semeval_test.csv")


if __name__ == '__main__':

    # log设置
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 训练和评估数据划分
    #train, val = train_test_split(train, test_size=.2)

    temp1 = list(map(str, train['text']))
    temp2 = list(map(str, val['text']))
    temp3 = list(map(str, test['text']))

    train_dict = {'label': train["label"], 'text': temp1}
    val_dict = {'label': val["label"], 'text': temp2}
    test_dict = {'text': temp3}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)


    # 加载分词器
    tokenizer = RobertaTokenizerFast.from_pretrained('xlm-roberta-base')


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # 加载模型
    model = RobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=1)

    # 加载评估方法
    metric = datasets.load_metric("pearsonr")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return metric.compute(predictions=predictions, references=labels)

    batch_size = 3
    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"

    )

    # early_stop = EarlyStoppingCallback(2, 1.0)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()



    prediction_outputs = trainer.predict(tokenized_test)

    #test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    test_pred = prediction_outputs.predictions[:, -1]
    print(test_pred)

    result_output = pd.DataFrame(data={"text": test["text"], "language": test["language"], "predictions": test_pred})
    result_output.to_csv("./result/xlm_roberta_new-triple_3_64.csv", index=False, quoting=3, escapechar='|')
    logging.info('result saved!')
