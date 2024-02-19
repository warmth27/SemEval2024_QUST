from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
    set_seed,
    LlamaTokenizer,
    LlamaForSequenceClassification,
)
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging

# from bstrai.classification_report import ClassificationReportModule
import torch


# torch.cuda.set_device(1)
# 输入文本预处理，将文本数据字段中的文本进行标记化（tokenization），
# 并且进行截断（truncation）处理
def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs["tokenizer"](
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


# 读取训练集和测试集
def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)  # 训练集
    test_df = pd.read_json(test_path, lines=True)  # 测试集

    # 将训练数据集划分为训练集train_df 和验证集val_df
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["label"], random_state=random_seed
    )

    return train_df, val_df, test_df


# 计算模型评估指标 F1
def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # np.argmax 函数，找到每个样本中预测概率最高的类别的索引

    results = {}
    results.update(
        f1_metric.compute(predictions=predictions, references=labels, average="micro")
    )

    return results


# 微调（fine-tune）预训练的文本分类模型
def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):
    # 将数据从 pandas dataframe 转换为 huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # 获取tokenizer和模型from huggingface    ### 需要修改 ###
    # tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    # model = AutoModelForSequenceClassification.from_pretrained(
    #    model, num_labels=len(label2id), id2label=id2label, label2id=label2id    # put your model here
    # )

    local_model_path = "/mnt/qust_521_big_2/public/deberta-v3-base/deberta-v3-large"
    # local_model_path = "/mnt/qust_521_big_2/public/llama-7b-hf"
    # local_model_path = '/mnt/qust_521_big_2/public/roberta/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )

    tokenized_valid_dataset = valid_dataset.map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path + "/best/"

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    trainer.save_model(best_model_path)
    print("saving best model ok....")


def test(test_df, model_path, id2label, label2id):
    # load tokenizer from saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # model = model.to('cuda:1')

    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("start to predict.....")
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    print("在softmax之前的logits:", predictions)
    # preds = predictions.predictions
    preds = predictions.predictions
    print("处理:", preds)
    return preds

    # print("start to predict..........")
    # # get logits from predictions and evaluate results using classification report
    # predictions = trainer.predict(tokenized_test_dataset)
    # prob_pred = softmax(predictions.predictions, axis=-1)
    # preds = np.argmax(predictions.predictions, axis=-1)
    # # metric = evaluate.load("bstrai/classification_report")   # 需要修改
    # # results = metric.compute(predictions=preds, references=predictions.label_ids)
    # metric = ClassificationReportModule()
    # results = metric.compute(predictions=preds, references=predictions.label_ids)
    # print(results)

    # # return dictionary of classification report
    # return results, preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 定义和解析命令行参数
    parser.add_argument(
        "--train_file_path",
        "-tr",
        required=True,
        help="Path to the train file.",
        type=str,
    )
    parser.add_argument(
        "--test_file_path", "-t", required=True, help="Path to the test file.", type=str
    )
    parser.add_argument(
        "--subtask",
        "-sb",
        required=True,
        help="Subtask (A or B).",
        type=str,
        choices=["A", "B"],
    )  # 用于指定子任务的标识符（A或B）
    parser.add_argument(
        "--model", "-m", required=True, help="Transformer to train and test", type=str
    )  # 用于指定要使用的Transformer模型的名称
    parser.add_argument(
        "--prediction_file_path",
        "-p",
        required=True,
        help="Path where to save the prediction file.",
        type=str,
    )  # 用于指定保存预测结果的文件路径。

    args = parser.parse_args()  # 解析命令行参数,存储在args对象

    random_seed = 0  # 设置随机种子
    train_path = args.train_file_path  # For example 'subtaskA_train_multilingual.jsonl'
    test_path = args.test_file_path  # For example 'subtaskA_test_multilingual.jsonl'
    model = args.model  # For example 'xlm-roberta-base'
    subtask = args.subtask  # For example 'A'
    prediction_path = (
        args.prediction_file_path
    )  # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    if subtask == "A":
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == "B":
        id2label = {
            0: "human",
            1: "chatGPT",
            2: "cohere",
            3: "davinci",
            4: "bloomz",
            5: "dolly",
        }
        label2id = {
            "human": 0,
            "chatGPT": 1,
            "cohere": 2,
            "davinci": 3,
            "bloomz": 4,
            "dolly": 5,
        }
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    # get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    # train detector model
    # fine_tune(
    #     train_df,
    #     valid_df,
    #     f"{model}/subtask{subtask}/{random_seed}",
    #     id2label,
    #     label2id,
    #     model,
    # )

    # test detector model
    # results, predictions = test(
    #     test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id
    # )

    # logging.info(results)
    # predictions_df = pd.DataFrame({"id": test_df["id"], "label": predictions})
    # predictions_df.to_json(prediction_path, lines=True, orient="records")

    predictions = test(
        test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id
    )

    # logging.info(results)
    predictions_df = pd.DataFrame(
        {"id": test_df["id"], "label": [item for item in predictions]}
    )
    predictions_df.to_json(prediction_path, lines=True, orient="records")






