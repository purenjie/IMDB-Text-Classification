import pandas as pd
import os

def loadDataset(data_dir):
    '''
    遍历训练集和测试集每个文件，读取内容并记录情感类别（正面/负面）
    Input：数据集地址
    Return：训练集和测试集数据；格式为 DataFrame，形式为 评论 类别
    '''
    data = {}
    count = 0
    for partition in ["train", "test"]:
        data[partition] = []
        for sentiment in ["neg", "pos"]:
            lable = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, partition, sentiment)
            files = os.listdir(path)
            for file in files:
                with open(os.path.join(path, file), "r") as f:
                    review = f.read()
                    data[partition].append([review, lable])
                count += 1
                print('读取%d个文件：' % count)
            
    
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'sentiment'])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'sentiment'])

    return data["train"], data["test"]

train_data, test_data = loadDataset('aclImdb')
train_data = pd.read_pickle('aclImdb/train_data.pkl')
print(train_data.info())
