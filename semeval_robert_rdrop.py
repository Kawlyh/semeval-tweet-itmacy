import os
import sys
import logging
import datasets
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.nn import MSELoss
from transformers import DataCollatorWithPadding, XLMRobertaPreTrainedModel, XLMRobertaModel, \
    XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification, get_cosine_schedule_with_warmup
from transformers import AdamW
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaClassificationHead

train = pd.read_csv("corpus/train.csv")
#train = pd.read_csv("corpus/new_triple_data.csv")
#val = pd.read_csv("corpus/new_val.csv")
test = pd.read_csv("corpus/semeval_test.csv")


class RoberScratch(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = XLMRobertaClassificationHead(config)

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    temp1 = list(map(str, train['text']))
    temp2 = list(map(str, val['text']))
    temp3 = list(map(str, test['text']))

    train_dict = {'label': train["label"], 'text': temp1}
    val_dict = {'label': val["label"], 'text': temp2}
    test_dict = {"text": temp3}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = RoberScratch.from_pretrained('xlm-roberta-base', num_labels=1)

    metric = datasets.load_metric("pearsonr")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    # steps = int(len(train["text"]) / training_args.per_device_train_batch_size)
    # warm_up_ratio = 0.1
    # optimizer = AdamW(model.parameters(), lr=0.001)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_training_steps=steps,
    #     num_warmup_steps=steps*warm_up_ratio,
    # )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, scheduler)
    )

    trainer.train()

    # 保存模型
    model.save_pretrained("checkpoint/my")

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = prediction_outputs.predictions[:, -1]
    print(test_pred)

    #result_output = pd.DataFrame(data={"text": test["text"], "language": test["language"], "predictions": test_pred})
    result_output = pd.DataFrame(data={"predictions": test_pred})
    result_output.to_csv("./result/final_xlm_roberta-triple_rdrop_epo-30_80.csv", index=False)
    logging.info('result saved!')
