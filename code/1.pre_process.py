#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pre_process.py
@Time    :   2020/08/03 20:45:10
@Author  :   Solejay 
@Version :   1.0
@Contact :   prj960827@gmail.com
@Desc    :   数据预处理，对影评数据进行清洗并绘制词云
             划分训练集和测试集，比例为7:3
'''
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
# 如果没有下载下面的工具需要取消注释下载
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def clean_data(data):
    '''
    对影评数据进行清洗
    输入：DataFrame 包含影评和情感类别
    输出：DataFrame 包含处理后的影评和数字化后类别
    '''
    # positive 用 1 替换，negative 用 0 替换
    data['sentiment'][data['sentiment']=='positive'] = 1
    data['sentiment'][data['sentiment']=='negative'] = 0

    for i, text in enumerate(tqdm(data['review'])):
        text = text.replace("<br />", " ") # 替换<br />标签
        lower = text.lower() # 全部转化为小写
        tokens = nltk.word_tokenize(lower) # 分词
        without_stopwords = [word for word in tokens if not word in stopwords.words('english')] # 去停用词
        no_alpha = [word for word in without_stopwords if word.isalpha()]
        wn = nltk.WordNetLemmatizer()
        lemm_text = [wn.lemmatize(word) for word in no_alpha] # 词形还原
        data['review'][i] = lemm_text
    
    return data

def processed_review(df, path):
    '''
    将清洗后的正面负面影评存储为 txt 格式
    输入：DataFrame 包含影评和情感类别, txt 存储位置
    '''
    sentiment = {0 : 'negative', 1 : 'positive'}
    for sent in sentiment:
        for review in tqdm(df[df['sentiment']==sent]['review']):
            txt_path = path + sentiment[sent] + '.txt' 
            with open(txt_path, 'a') as f:
                f.write(review + '\n')

def draw_word_cloud(txt_path):
    '''
    接收文本路径并绘制词云
    输入：txt 文件路径
    '''
    content = open(txt_path,'r',encoding='utf-8').read()
    
    # 绘制词云
    #生成一个词云对象
    wordcloud = WordCloud(
            background_color="white", #设置背景为白色，默认为黑色
            width=1500,              #设置图片的宽度
            height=960,              #设置图片的高度
            margin=10,               #设置图片的边缘
            ).generate(content)
    # 绘制图片
    plt.imshow(wordcloud)
    # 消除坐标轴
    plt.axis("off")
    # 展示图片
    plt.show()
    # 保存图片
    wordcloud.to_file('../data/wordcloud/%s.png' % txt_path[18:-4])

def split_data(data):
    """
    对数据进行随机划分，训练集和测试集的比例为7:3
    """
    x_train, x_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.3, random_state=42)
    with open('../data/split_dataset.pkl', 'wb') as f:  
        pickle.dump([x_train, x_test, y_train, y_test], f)


# 数据清洗过程
data= pd.read_csv('../data/IMDB Dataset.csv')
clean_data = clean_data(data)
clean_data['review'] = [' '.join(sent) for sent in clean_data['review']]
clean_data.to_csv('../data/clean_dataset.csv', columns=['review', 'sentiment'], index=False)

# 绘制词云过程
clean_data = pd.read_csv('../data/clean_dataset.csv')
processed_review(clean_data, '../data/wordcloud/')
draw_word_cloud('../data/wordcloud/positive.txt')
draw_word_cloud('../data/wordcloud/negative.txt')

# 划分训练集和测试集过程
clean_data = pd.read_csv('../data/clean_dataset.csv')
split_data(clean_data)
