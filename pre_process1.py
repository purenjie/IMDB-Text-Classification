import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
            str = ' '.join(review)
            with open(txt_path, 'a') as f:
                f.write(str + '\n')

def draw_word_cloud(txt_path):
    '''
    接收文本路径并绘制词云
    输入：txt 文件路径
    '''
    use_NN = False

    if use_NN:
        # 读取内容
        with open(txt_path,'r',encoding='utf-8') as f:
            content = f.read()

        words = nltk.word_tokenize(content) # 分词
        tags = set(['NN', 'NNS', 'NNP', 'NNPS']) # 名词
        pos_tags =nltk.pos_tag(words)
        tmp = []
        for word,pos in pos_tags:
            if (pos not in tags):
                tmp.append(word)
        content = ' '.join(tmp)
        with open('data/neg.txt', 'a') as f:
            f.write(content)
    else:
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
    wordcloud.to_file('data/%s.png' % txt_path[5:13])



    

# data= pd.read_csv('data/IMDB Dataset.csv')
# clean_data = clean_data(data)
# clean_data.to_csv('data/clean_dataset.csv', columns=['review', 'sentiment'], index=False)

# clean_data = pd.read_pickle('data/clean_data.pkl')
# processed_review(clean_data, 'data/')

draw_word_cloud('data/negative.txt')
