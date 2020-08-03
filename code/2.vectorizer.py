from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pandas as pd
import pickle






# 读取训练集和测试集数据
with open('data/processed_dataset.pkl', 'rb') as f: 
    x_train, x_test, y_train, y_test = pickle.load(f)

bow_vect = CountVectorizer()
# 保存中间结果，⽅便后⾯使⽤TfidfTransformer
imdb_data_bow = bow_vect.fit_transform(imdb_data['clean_text'])
# 转换训练集，验证集，测试集
X_train_bow = bow_vect.transform(X_train_text)
X_val_bow = bow_vect.transform(X_val_text)
X_test_bow = bow_vect.transform(X_test_text)