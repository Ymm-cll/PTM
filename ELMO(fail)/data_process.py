import pandas as pd
from sklearn.model_selection import train_test_split
import json

def txt_to_tsv():
    input_file = "../data/raw/imdb_test.txt"
    output_file = "../data/raw/imdb_test.tsv"

    data = pd.read_csv(input_file, sep=",", header=None, names=["id", "comment", "label"])

    # 删除多余的引号和逗号
    data["id"] = data["id"].replace('"', "")
    data["label"] = data["label"].replace('"', "")
    data["comment"] = data["comment"].replace(',"', "\t")

    data.to_csv(output_file, sep="\t", index=False)

def to_one():
    train_file = "../data/raw/imdb_train.tsv"
    test_file = "../data/raw/imdb_test.tsv"
    output_file = "../data/raw/imdb.tsv"

    # 读取训练集和测试集
    train_data = pd.read_csv(train_file, sep="\t")
    test_data = pd.read_csv(test_file, sep="\t")

    # 合并训练集和测试集
    combined_data = pd.concat([train_data, test_data], axis=0)

    # 保存合并后的数据集为TSV文件
    combined_data.to_csv(output_file, sep="\t", index=False)

def rearrange():
    input_file = "../data/raw/imdb.tsv"
    train_output_file = "../data/rearranged/imdb_train.tsv"
    val_output_file = "../data/rearranged/imdb_val.tsv"
    test_output_file = "../data/rearranged/imdb_test.tsv"

    # 读取IMDb数据集
    data = pd.read_csv(input_file, sep="\t")

    # 划分训练集和剩余数据
    train_data, remaining_data = train_test_split(data, test_size=0.4, random_state=42)

    # 进一步划分剩余数据为验证集和测试集
    val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42)

    # 保存划分后的数据集为TSV文件
    train_data.to_csv(train_output_file, sep="\t", index=False)
    val_data.to_csv(val_output_file, sep="\t", index=False)
    test_data.to_csv(test_output_file, sep="\t", index=False)

def tsv_to_json():
    # 读取 IMDb 数据集的 TSV 文件
    data = pd.read_csv("../data/rearranged/imdb_val.tsv", sep="\t", header=None, names=["text", "label"])

    # 将数据转换为 JSON
    data_json = data.to_json(orient="records")

    # 将 JSON 写入文件
    with open("../data/rearranged/imdb_val.json", "w") as json_file:
        json_file.write(data_json)

def json_to_jsonl():
    # 读取JSON文件
    with open('../data/rearranged/imdb_test.json', 'r') as json_file:
        data = json.load(json_file)

    # 转换为JSONL格式并写入文件
    with open('../data/rearranged/imdb_test.jsonl', 'w') as jsonl_file:
        for item in data:
            jsonl_string = json.dumps(item) + '\n'
            jsonl_file.write(jsonl_string)

# txt_to_tsv
# to_one()
# rearrange()
#tsv_to_json()
json_to_jsonl()