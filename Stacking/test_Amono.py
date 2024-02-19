import torch
import torch.nn as nn
import torch.optim as optim
import json

torch.manual_seed(0)


# 读取JSONL文件并提取logits的函数
def read_logits_from_jsonl(file_name):
    logits_list = []
    with open(file_name, "r") as file:
        for line in file:
            data = json.loads(line)
            logits = torch.tensor(data["label"])
            logits_list.append(logits)
    return torch.stack(logits_list)


# 读取JSONL文件并提取标签的函数
def read_labels_from_jsonl(file_name):
    labels = []
    with open(file_name, "r") as file:
        for line in file:
            data = json.loads(line)
            label = data["label"]
            labels.append(label)
    return torch.tensor(labels)


# 定义模型
class Adapter_Origin(nn.Module):
    def __init__(self):
        super(Adapter_Origin, self).__init__()
        self.fc = nn.Linear(2 * 2, 2)

    def forward(self, logits1, logits2):
        combined_logits = torch.cat((logits1, logits2), dim=1)
        return self.fc(combined_logits)


# 实例化模型
adapter = Adapter_Origin()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(adapter.parameters(), lr=0.001)


# 读取logits数据和标签
train_logits1 = read_logits_from_jsonl(
    "/home/qd/PycharmProjects/SemEval_Test/A_Voting/logits/lxr_train_logits.jsonl"
)  # 李祥润
train_logits2 = read_logits_from_jsonl(
    "/home/qd/PycharmProjects/SemEval_Test/SubtaskA_deberta/deberta_A/A_deberta_train_logits.jsonl"  # 汪洋
)
train_labels = read_labels_from_jsonl(
    "/home/qd/PycharmProjects/semeval_lxr/SemEval2024-task8-main/SemEval2024-task8-main/subtaskB/scorer/train_gold.jsonl"
)

# train_logits1 = read_logits_from_jsonl(
#     "/home/qd/PycharmProjects/SemEval_Test/A_Voting/logits/lxr_dev89_logits.jsonl"
# )  # 李祥润
# train_logits2 = read_logits_from_jsonl(
#     "/home/qd/PycharmProjects/SemEval_Test/A_Voting/logits/dev84_logits.jsonl"  # 汪洋
# )
# train_labels = read_labels_from_jsonl(
#     "/home/qd/PycharmProjects/semeval_lxr/gold_train_multilingual.jsonl"
# )
# train_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = adapter(train_logits1, train_logits2)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

print("start to predict.....")

# 读取预测数据
new_logits1 = read_logits_from_jsonl(
    "/home/qd/PycharmProjects/SemEval_Test/AAB-Test/A-mono-test/lxr_Amon_test_logits_process.jsonl"
)  # LXR
new_logits2 = read_logits_from_jsonl(
    "/home/qd/PycharmProjects/SemEval_Test/AAB-Test/A-mono-test/A_mono_deberta_test_logits_pro.jsonl"  # WY
)

# 使用训练好的模型进行预测
with torch.no_grad():
    model_predictions = adapter(new_logits1, new_logits2)
    predicted_classes = torch.argmax(model_predictions, dim=1)


# 将预测结果保存到JSONL文件的函数
def save_predictions_to_jsonl(predictions, file_name, input_jsonl_file_name):
    with open(input_jsonl_file_name, "r") as input_file:
        lines = input_file.readlines()

    with open(file_name, "w") as file:
        for idx, label in enumerate(predictions):
            data = {"id": json.loads(lines[idx])["id"], "label": label.item()}
            json_line = json.dumps(data)
            file.write(json_line + "\n")


# 保存预测结果
save_predictions_to_jsonl(
    predicted_classes,
    "subtask_a_mono_pro2.jsonl",
    "/home/qd/PycharmProjects/SemEval_Test/AAB-Test/A-mono-test/A_mono_deberta_test_logits_pro.jsonl",
)
print("saving predictions ok!!!!!")
