import json


# 读取JSONL文件并将其转换为JSON
def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        jsonl_data = f.readlines()

    json_data = [json.loads(line.strip()) for line in jsonl_data]

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def josn_to_csv():
    import csv
    import json
    # 从JSON文件中读取数据
    with open(r'E:\pythonwork\NLP\Semeval\SubtaskA\SubtaskA\subtaskA_train_multilingual.json', 'r',
              encoding='utf-8') as json_file:
        data = json.load(json_file)

    # 设置CSV文件名和列名
    csv_filename = 'subtaskA_train_multilingual.csv'
    csv_columns = ['text', 'label', 'model', 'source', 'id']  # 请根据您的JSON数据结构更改列名

    # 将JSON数据写入CSV文件
    with open(csv_filename, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()  # 写入列名

        for item in data:
            writer.writerow(item)

    print(f'成功将JSON数据转换为CSV文件：{csv_filename}')


if __name__ == "__main__":
    # jsonl_file = "SubtaskA\SubtaskA\subtaskA_train_multilingual.jsonl"  # 输入的JSONL文件名
    # json_file = "SubtaskA\SubtaskA\subtaskA_train_multilingual.json"   # 转换后的JSON文件名
    #
    # jsonl_to_json(jsonl_file, json_file)
    josn_to_csv()
