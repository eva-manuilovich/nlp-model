#General
import pandas as pd
import numpy as np
import os
import re
import json
import string
import time
import math
import requests
from scipy import sparse
from datetime import datetime, date
from tqdm.notebook import tqdm
import functools
from functools import reduce

#Text
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
punct = string.punctuation + '«»'
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from pymystem3 import Mystem #оч клевая лемматизация от Яндекса
m = Mystem()

#Modeling
from sklearn.decomposition import NMF
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

#загрузка данных из файла, использовавшегося 
#в неактуальной бибилиотеке djantimat
bad_words = []
with open('initial_data.json') as json_file:
    data = json.load(json_file)
    for p in data:
        bad_words.append(p['fields']['word'])
        
vk_stopwords = ['контент', 'информация', 'вконтакте', 'сообщение', 'пост', 'стена', 'обсуждение',
                'подписчик','страница','сайт','группа','официальный', 'ся', 'сообщество' 'пост',
                'правило','комментарий', 'участник', 'администрация', 'паблик', 
                'пользователь', 'ссылка','предложка', 'понедельник','вторник',
                'среда','четверг','пятница','суббота','воскресенье','админ', 'вс', 'сво', 'укр',
                'наш', 'ул', 'свой', 'это', 'который', 'самый', 'год', 'неделя', 'день', 'ваш']

geo = pd.read_csv('geo_rus.csv')

def date_to_age(bdate):
    bdate = str(bdate)
    if len(bdate.split('.'))==3:
        d, m, y = bdate.split('.')
        d = int(d)
        m = int(m)
        y = int(y)
        try:
            age = int((datetime.date.today()-datetime.date(day=d, month=m, year=y)).days/365.25)
        except:
            age = np.nan
    else:
        age = np.nan
    return age
    
def broad_posts(x, cat, broad_posts_words):
    for word in broad_posts_words[cat]:
        if word in x:
            return False
    else:
        return True

def replace_semicolon(x):
    if isinstance(x, str):
        return x.replace(';', ",").replace('\r', ' ')
    else:
        return x
def remove_stws(x, stws):
    stws = '|'.join([f'\\b{j}\\b' for j in stws])
    x = re.sub(f'{stws}', '', x)
    return x
    
def first_process(df, concat_parent_post=1):
    """Производит первичную обработку датасета"""
    df = df[df['fullText'].notna()]
    df = df.applymap(replace_semicolon) #уберем из текстов точку с запятой, чтобы не было проблем при сохранении
    df = df[df['autoCategories'].astype(str).apply(lambda x: 'wom' in x)] #отберем посты пользователей
    df = df[df['author.gender']!='community'] #опять же, только людей, но не сообществ
    df['author.age'] = df['author.age'].apply(lambda x: np.nan if x>90 else x)
    df = df.drop_duplicates(subset=['fullText']) 
    df['fullText_lemm'] = lemm_text(df['fullText']) #создадим столбец с лемматизированными постами    
    df['vk_id'] = df['author.url'].apply(vk_id)
    
    df['parentPostId'] = df['parentPostId'].fillna('')
    df['textWithParent'] = df['fullText']
    if concat_parent_post==1:
        print('Присоединение родительских постов')        
        for k in tqdm(range(len(df)//10000)):
            df_part = df[k*10000:(k+2)*10000]
            for i in df_part.index:
                if (df_part.loc[i, 'parentPostId']!=''):
                    if df_part.loc[i, 'parentPostId'] in list(df_part['postId']):
                        df.loc[i, 'textWithParent'] = df_part['fullText'][
                            df_part['postId']==df_part.loc[i, 'parentPostId']].values[0] + ' ' +df_part.loc[i, 'fullText']
        
    return df

def delete_unwanted_mentions(df, word_list):
    """Удаляем посты, в которых встречаются токены, однозначно сигнализируещие о неревантных упоминаниях"""
    if 'fullText_lemm' not in list(df.columns):
        df['fullText_lemm'] = lemm_text(df['fullText'])
    if len(word_list)>0:
        df = df[~df['fullText'].str.contains('|'.join(word_list), case=False, regex=True).astype(bool)]
        df = df[~df['fullText_lemm'].str.contains('|'.join(word_list), case=False, regex=True).astype(bool)]
    return df
 

def cross_posting_remove(df, min_words=10):
    """Функция, убирающая посты, подозрительно друг на друга похожие."""
    ind_list = []
    #Похожие посты обычно находятся рядом, поэтому будем рассматривать куски по 10к длиной
    #С полным перекрытием
    #т.е. берем посты с 0 по 20к, затем с 10к по 30к и т.д.
    #Брать все сразу лучше не стоит, так как метод включает в себя подсчет матрицы косинусных расстояний,
    #объем которой растет с квадратичной скоростью от длины датасета
    if 'fullText_lemm' not in list(df.columns):
        df['fullText_lemm'] = lemm_text(df['fullText'])
    for i in tqdm(range(len(df)//10000)):
        df_part = df[i*10000:(i+2)*10000]
        #Среди коротких постов нет кросспостинга
        #10 слов для min_words - порог, выбранный на основе тестирования vk
        #для других соцсетей он может быть иным
        df_cut = df_part[df_part['fullText'].apply(lambda x: len(x.split())>min_words)]
        tfidf = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 1))
        X = tfidf.fit_transform(df_cut['fullText_lemm'].to_list())
        voc = pd.Series(tfidf.vocabulary_).sort_values().index
        cs = pd.DataFrame(cosine_similarity(X))
        ind_list = ind_list + list(df_cut.iloc[cs[(cs>0.9).astype(int).sum()>1].index].index)
    df = df[~df.index.to_series().isin(list(set(ind_list)))]
    return df

"""Функции для построения тематик"""
def category_search(x, category_tokens):
    """отбор упоминаний, имеющих отношение к категории"""
    if isinstance(category_tokens, list) and isinstance(category_tokens[0], str):
        return bool(re.search('|'.join(category_tokens).lower(), x.lower()))
    else:
        return True
    
def find_brands(x, cat_brand_dict):
    """посик упоминаний брендов в постах"""
    x = re.sub(r'[^\w\s]', ' ', x)
    x = re.sub(r' +', ' ', x)
    x = x.lower() 
    brands_contained = set()
    for brands in cat_brand_dict:
        for brand in brands:
            if re.search('\\b'+brand.lower(), x) is not None:
                brands_contained.add(brands[0])
    return list(brands_contained)

def brand_columns(df, cat_brand_dict):
    """Создает в таблице бинарные колонки под каждый бренд и возвращает таблицу и их список"""
    #мы хотим искать названия брендов не только в самом посте, но и в родительском
    df['brands'] = (df['textWithParent'].astype(str)+' '+df['fullText_lemm'].astype(str)).apply(lambda x: find_brands(x, cat_brand_dict))
    #создадим колонки с брендами
    brand_cols = []
    for brands in cat_brand_dict:
        brand_cols.append(brands[0])
        df[brands[0]] = 0
        df[brands[0]] = df['brands'].apply(lambda x: 1 if brands[0] in x else 0)
    return (df, brand_cols)

def separate_brands(df, cat_brand_dict, brand_cols):
    """
    Разделяет посты, содержащие упоминания нескольких брендов по знакам препинания.
    ---------
    Input
    df: DataFrame, исходная таблица с неразделенными постами
    category: str, категория товаров
    brand_cols: list of str, список названий бинарных столбцов брендов
    Output
    df: DataFrame, таблица с разделенными брендами
    """
    new_lines = []
    to_drop = []
    for line in df.index:
        #выбираем строки, где брендов больше одного
        if df.loc[line, :][brand_cols].sum()>1:
            init_text = df.loc[line, :]['textWithParent']
            init_text_and_lemm = init_text + ' ' + str(df.loc[line, :]['fullText_lemm'])
            brands_in_post = find_brands(init_text_and_lemm, cat_brand_dict)

            #объединим все названия брендов категории вместе, чтобы искать их в строке
            b_list = []
            for b in brands_in_post:
                b_list.append('|'.join(['\\b'+word for word in [l for l in cat_brand_dict if b in l][0]]).lower())
            b_list = '|'.join(b_list)
            #я люблю регулярные выражения...

            #Уберем знаки препиная из всей первой части до первого упоминания любого бренда
            union_list = ['или', 'но', 'а', 'хотя']
            union_list = ['\\b'+u+'\\b' for u in union_list]
            
            init_text = re.sub(f'[.,!?;]|{"|".join(union_list)}|\\n', ' ', re.split(
                f'({b_list})', init_text.lower())[0]) + ' '.join(
                                                    re.split(f'({b_list})', init_text.lower())[1:])
            #разделим строку по знакам препинания, после которых до следующего знака препинания идет название брнеда
            texts = re.split(f'(?:[.,!?;]|{"|".join(union_list)}|\\n)(?=(?:(?![.,!?;]|{"|".join(union_list)}|\\n).)*(?:{b_list}))', init_text.lower())

            #мы хоти объединить утверждения с одинаковыми брендами вместе
            united_texts = {}
            united_texts_lemm = {}
            for text in texts:
                text_lemm = re.sub('\\n', '', ' '.join(m.lemmatize(text))).strip()
                text_brands = str(find_brands(text+' '+text_lemm, cat_brand_dict))  
                
                if text_brands in list(united_texts.keys()):
                    united_texts[text_brands] += ' '+text
                    united_texts_lemm[text_brands] += ' '+text_lemm
                else:
                    united_texts[text_brands] = text
                    united_texts_lemm[text_brands] = text_lemm

            #а теперь создадим отдельные строки на каждую часть утверждения        
            for brand in united_texts:
                new_line = df.loc[line, :]
                new_line['textWithParent'] = united_texts[brand]
                new_line['fullText_lemm'] = united_texts_lemm[brand]
                new_lines.append(new_line)
            to_drop.append(line)
    #и убрать строки с брендами вместе
    df = df.drop(to_drop, axis=0)
    df['multibrand'] = 0
    new_df = pd.DataFrame(new_lines).reset_index(drop=True)
    new_df['multibrand'] = 1
    df = pd.concat([df, new_df], sort=False).reset_index(drop=True)

    #восстановим правильные бренды (пока ленивым способом, просто пересчитав)
    df['brands'] = (df['textWithParent'].astype(str)+' '+df['fullText_lemm'].astype(str)).apply(lambda x: find_brands(x, cat_brand_dict))
    #создадим колонки с брендами
    brand_cols = []
    for brands in cat_brand_dict:
        brand_cols.append(brands[0])
        df[brands[0]] = 0
        df[brands[0]] = df['brands'].apply(lambda x: 1 if brands[0] in x else 0)
    #Чтобы проще было их определить в дашборде
    df = df.rename(columns=dict(zip(brand_cols, [col+'_brand' for col in brand_cols])))
        
    return df

def preprocess_text(text, stopwords):
    """для очистки поля fullText от всего, что не слова и стоп-слов"""
    text = re.sub('(((?![ёЁа-яА-Я ]).)+)', ' ', text)
    text = re.sub(' +', ' ', text).lower() 
    tokens = [w for w in text.split() if w not in stopwords]
    return ' '.join(tokens)  

def lemm_text(col_in, stopwords=russian_stopwords):
    """производит лемматизацию текста в переданном столбце"""
    col_lemm = col_in.astype(str).apply(lambda x: preprocess_text(x, stopwords))
    def clean_text(x):
        try:
            #почему-то после лемматизации остаются переносы строки
            x = re.sub('\\n', '', ' '.join(m.lemmatize(x))).strip()
            x = ' '.join([w for w in x.split() if w not in stopwords])
            #нас таки интересуют отрицания
            x =  x.lower().replace(' не ', ' не_').replace(' без ', ' без_')
        except:
            x = ''
        return x
    col_lemm = col_lemm.apply(clean_text)
    return col_lemm

def multibrand_sentimet(X, y, multi):
    """Проставляет тональность у кусочков мультибредовых постов, обучаясь на однобрендовых"""
    train_mask = ((multi==0)&((y=='positive')|(y=='negative'))).to_list()
    X_train = X.A[train_mask, :]
    y_train = y[train_mask]
    y_train = (y_train=='positive').astype(int)
    X_test = X.A[(multi==1).to_list(), :]
    
    clf = RandomForestClassifier(n_estimators=150, class_weight='balanced')
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
        sent_dict = {
            0: 'negative',
            1: 'positive'
        }
        y_pred = [sent_dict[sent] for sent in y_pred]
        y[(multi==1).to_list()] = y_pred 
    except:
        print('увы, не получилось перерассчитать тональность')
    return y

def find_similar_topics(base_model, models_list, th=0.7):
    """
    Поиск близких тематик на основе косинусного расстояния.
    --------------------------
    Input
    base_model: tuple, из двух элементов, включиет в себя базовую матрицу компонент тематик и базовую матрицу постов, раскиданных по тематикам
    models_list: list of tuples, те же самые, что и в base_model, пары, но являющиеся основном массой ансамбля 
    th: float, (0, 1), порог объединения тематик
    Output
    tuple, такой же, как в base model, но являющиеся объединением моделей ансамбля
    """
    agg_topics, agg_x = base_model
    base_topics = agg_topics.copy()
    for base in base_topics.index:
        base_topic = base_topics.loc[base, :]
        sim_count = 1
        for model in models_list:
            topics, x = model
            for other in topics.index:
                sim = cosine_similarity([topics.loc[other, :]], [base_topic])
                if sim>th:
                    sim_count += 1
                    agg_topics.iloc[base, :] = agg_topics.loc[base, :] + topics.loc[other, :]
                    agg_x.loc[:, base] = agg_x.loc[:, base] + x.loc[:, other]
        agg_topics.loc[base, :] = agg_topics.loc[base, :]/sim_count
        agg_x.loc[:, base] = agg_x.loc[:, base]/sim_count        
    return (agg_topics, agg_x)

def modeling(X, power, th, t_num, voc, forest=15):
    """
    Разделяет комментарии на тематики
    --------
    Input
    X: 2d array or DataFrame, векторизованные тексты
    power: float, степень, в которую нужно возвести каждый элемент итоговой матрицы вероятностей 
        для нормировки суммы вероятностей на единицу
    th: float, порог косинусной близости, начиная с которого объединяются тематики
    t_num: int, количество тематик, на которые нужно бдет разделить тексты
    voc: list of str, названия столбцов, полученные в результате векторизации текстов
    Output
    topics: DataFrame, таблица, содержащая веса слов в тематиках (кол-во тематик * кол-во слов)
    x: DataFrame, таблица, содержащая веса постов в тематиках (кол-во постов * кол-во тематик)
    """  
    first_seed = 57 #для всопроизводимости результатов
    nmf_models = []
    nmf_brands = []
    for i in range(forest):
        nmf = NMF(t_num+i//5, random_state=first_seed+i)
        nmf_x = nmf.fit_transform(X)
        nmf_x = pd.DataFrame(nmf_x)

        nmf_topics = nmf.components_
        nmf_topics = pd.DataFrame(nmf_topics, columns=voc)
        nmf_topics = (nmf_topics.T / nmf_topics.sum(axis=1)).T
        nmf_models.append((nmf_topics, nmf_x))
    topics, x = find_similar_topics(nmf_models[-1], nmf_models[:-1], th)
    #Возведение каждой вероятности в подсчитанную ранее степень 
    #для нормировки суммарной вероятности на единицу
    x = x.applymap(lambda x: math.pow(x, power))
    return (topics, x)

def P_w(w_list, docs):
    """
    функция для подсчета вероятности встретить одно 
    слово или несколько слов вместе в текстах docs
    """
    if isinstance(w_list, str)|(len(w_list)==1):
        w = w_list
        #поправка на случай равенства нулю итогового количества
        p = 0
        for doc in docs:
            if len(re.findall('\\b'+w+'\\b', doc))>0:
                p += 1
        return p/len(docs)
    else:
        #поправка на случай равенства нулю итогового количества
        p = 0
        for doc in docs:
            p = p + int(functools.reduce((lambda a, b: a&b), [len(re.findall(f'\\b{x}\\b', doc))>0 for x in w_list]))
        return p/len(docs)
                
def coherent_measure(topic, docs):
    """подсчет когерентности"""
    wco = []
    for i, w1 in enumerate(topic.split()):
        for w2 in topic.split():
            if w1!=w2:
                p12 = P_w([w1, w2], docs)                    
                p1 = P_w(w1, docs)
                if p1!=0:
                    wco.append(np.log2(1 + p12/p1))
    try:
        wco = np.mean(wco) 
    except:
        wco = 0
    return wco

def create_namings(topics, all_names):
    """
    Находит название с ближайшим описанием (по встерчаемости слов)
    --------
    Input
    topics: list of str, список тематик, представляющих из себя соединенные в строку ключевые слова
    all_names: dict of str:str, избыточный словарь (на все случаи жизни, я надеюсь), содержащий названия тематик и дополнительные описания
    Output
    renaming_dict: dict of str:str, словарь для переименования столбцов в общей таблице
    """
    renaming_dict = {}
    for topic in topics:
        max_count = 0
        naming = ''
        for name in all_names:
            count = 0
            #дополнительные описания - это случайно набросанные куски от отзывов, 
            #поэтому там встречаются случайные символы, слова в любых формах и тэдэ
            name_descr = (' '.join(m.lemmatize(all_names[name]))).lower()
            name_descr = name_descr.replace('не ', 'не_').replace('ни ', 'ни_').replace('без ', 'без_')
            name_descr = name_descr.split()
            #просто посчитаем, сколько слов из тематики есть в этом описании
            for w in topic.split():
                if w in name_descr:
                    count += 1
#             count = count/len(topic.split())
            count_frac = count/len(name_descr)
            if (count_frac>max_count)&(count>=1):
                max_count = count_frac
                naming = name
        renaming_dict[topic] = naming
    return renaming_dict

def rename_topics(df, topics_naming_dict):
    """Убирает тематики без названия, переименовывает и объединяет одинаковые"""
    to_drop = [topic for topic in topics_naming_dict if topics_naming_dict[topic]=='']
    df = df.drop(columns=to_drop)
    df = df.rename(columns=topics_naming_dict)
    df = df.sum(axis=1, level=0)
    new_names = [t for t in list(topics_naming_dict.values()) if t!='']
    #где-то могло стать больше 1
    df[new_names] = df[new_names].applymap(lambda x: x if x<1 else 1)
    return df

def build_topics(df_init, category, theme_params, separ_brands=True, sentiment_list=['positive', 'negative', 'neutral'], category_tokens={}, multibrand_sent=True, drop_multibrand=False, add_columns=[], names_dict=None, forest=10, params=(5, 0.5, 0.15, 10), th=0.7, power=1):
    """
    Объединяет в себе весь путь подготовки таблицы для дашборда
    --------
    Input
    df: DataFrame, таблица, выгруженная из YouScan (единственное что, с присоединенные родительскими постами), 
    category: str, название котегории товаров, 
    theme_params: tuple (list of str, dict of lists) 
                theme_stopwords: специфические для темы стоп-слова
                brand_dict: словарь категорий-брендов, где прописаны все наименования, относящиеся к категории и их возможные написания
    separ_brands: bool, флаг, указывающий, надо ли разделять на части посты с несколькими брендами внутри 
    sentiment_list: list of str, список тональностей, включаемых в анализ
    category_tokens: list of str, Список токенов, котрые определяют принадлежность к категоии. в случае передачи пустого списк, берется список написаний брендов
    multibrand_sent: bool, флаг, указывающий, надо ли пересчитывать тональность у разделенных постов
    add_columns: list of str, список колонок из датасета, которые нужно включить в датасет,
                            сейчас поддерживаются колонки 'author.age' и 'author.gender'
    names_dict: dict, словарь категории-(тональность-(темы-ключевые слова)), который позволяет присваивать тематикам ёмкие короткие названия
    forest: int, число, определяющее количество моделей в ансамбле
    params: tuple (float, float, int), набор параметров (tfidf_max_df, coherence_th, t_num) 
            tfidf_max_df: по какой доле встречаемости обрезать словарь векторизации
            coherence_th: начиная с какого значения когерентности тематики пропускать ее в финальный этап
            t_num: минимальное количество тематик в моделе 
    th: float, порог косинусной близости, начиная с которого объединяются тематики
    power: float (0, 1], параметр, указывающий, в какую степень нужно возводить результаты моделирования, 
                        чтобы получить выборку с правильной средней суммой вероятностей по тематикам,
                        считается с помощью функции power_calculating
    Output
    final_df: DataFrame, итоговая таблица с заданными категорией и тональностью
    """
    df = df_init.copy()   
    theme_stopwords, brand_dict = theme_params
    
    tfidf_min_df, tfidf_max_df, coherence_th, t_num = params
    
    if category_tokens=={}:
        category_tokens = functools.reduce((lambda a, b: a+b), brand_dict[category])
    else:
        category_tokens = category_tokens[category]
    
    if 'textWithParent' not in list(df.columns):
        df['textWithParent'] = df['fullText']
    
    #выберем, что отфильтровывать
    
    wom = (df['autoCategories'].apply(lambda x: 'wom' in x))
    cat = df['textWithParent'].apply(lambda x: category_search(x, category_tokens))
    
    df = df[wom&cat].drop_duplicates(subset=['textWithParent'])
    df['Category'] = category
    
    stws = list(russian_stopwords).copy()
    stws = [w for w in stws if w not in ['без', 'не']]
    
    df = df[df['textWithParent'].notna()]
    
    if ('fullText_lemm' not in list(df.columns))|(separ_brands):
        df['fullText_lemm'] = lemm_text(df['textWithParent'], stws+vk_stopwords+theme_stopwords)
    else:
        df['fullText_lemm'] = df['fullText_lemm'].apply(lambda x: remove_stws(x, stws+vk_stopwords+theme_stopwords))

    df, brand_cols = brand_columns(df, brand_dict[category])
    if drop_multibrand:
        df = df[df['brands'].apply(lambda x: len(x)==1)]
    if separ_brands:        
        df = separate_brands(df, brand_dict[category], brand_cols) 
    df = df.rename(columns=dict(zip(brand_cols, [col+'_brand' for col in brand_cols])))
    df['brands'] = df['brands'].astype(str)
    df = df.drop_duplicates(subset=['textWithParent', 'brands'])
        
    #а еще мы не хотим названий брендов в постах вообще
    all_brand_or = '|'.join([f'\\b{j}(?:[ауе]|ом|ем|ой)?\\b' for i in brand_dict[category] for j in i]).lower()
    df['fullText_lemm'] = df['fullText_lemm'].apply(lambda x: re.sub(f'{all_brand_or}', '', x))
    
    #ограничим посты по количеству содержательных слов
    len_mask = df['fullText_lemm'].apply(lambda doc: (len(doc.split())>3)&(len(doc)<700)).to_list()
    df = df[len_mask]
    docs = df['fullText_lemm'].to_list()

    #векторизуем тексты
    tfidf = TfidfVectorizer(min_df=tfidf_min_df, max_df=tfidf_max_df, ngram_range=(1, 1))
    X = tfidf.fit_transform(docs)
    
    #запомним, какая координата какое слово означает
    voc = pd.Series(tfidf.vocabulary_).sort_values().index
    if multibrand_sent&separ_brands:
        df['sentiment'] = multibrand_sentimet(X, df['sentiment'], df['multibrand'])
        
    if len(add_columns)>0:
        for col in add_columns:
            if col in list(df.columns):
                if col=='author.age':
                    bins = pd.IntervalIndex.from_tuples([(0, 23), (23, 30), (30, 40), (40, 50), (50, 60), (60, 2000)], closed='left')
                    labels = dict(zip(['[0.0, 23.0)', '[23.0, 30.0)', '[30.0, 40.0)', '[40.0, 50.0)', '[50.0, 60.0)', '[60.0, 2000.0)', 'nan'],
                      ['_Возраст 0-23_', '_Возраст 23-30_', '_Возраст 30-40_', '_Возраст 40-50_', '_Возраст 50-60_', '_Возраст >60_', np.nan]))                  
                    df['age_groups'] = pd.cut(df['author.age'].replace('-', np.nan).astype(float), bins).astype(str).replace(labels)
                    age_columns = pd.get_dummies(df['age_groups'], prefix='', prefix_sep='')
                    X = sparse.hstack([X, sparse.csr_matrix(age_columns)])
                    voc = list(voc)+list(age_columns.columns)
                elif col=='author.gender':
                    df['author.gender'] = df['author.gender'].replace({'male': 1, 'female': 0})
                    gender_columns = pd.get_dummies(df['author.gender'], prefix='', prefix_sep='')
                    gender_columns = gender_columns.rename(columns={'0': '_Женщина_', '1': '_Мужчина_'})
                    X = sparse.hstack([X, sparse.csr_matrix(gender_columns)])
                    voc = list(voc)+list(gender_columns.columns)
    
    final_df = []
    for sentiment in sentiment_list:
        X_sent = X.A[df['sentiment']==sentiment, :]
        df_sent = df[df['sentiment']==sentiment]
        
        #создадим тематики
        topics, x = modeling(X_sent, power, th, t_num, voc, forest)    

        #отберем топ ключевых слов по тематикам
        result_df = []
        for i in topics.index:
            words = list(topics.loc[i, :].sort_values(ascending=False).head(15).index)
            result_df.append(words)
        result_df = pd.DataFrame(result_df)

        #посчитаем для них когерентность
        topic_list = []
        for i in result_df.index:
            topic = ' '.join(result_df.loc[i, :].to_list())
            measure =  coherent_measure(topic, docs)
            topic_list.append((topic, measure))

        #возьмем только прошедшие отбор
        topics_mask = [t[1]>coherence_th for t in topic_list] 
        x.columns = [el[0]+'_'+str(el[1]) for el in topic_list]
        x = x[x.columns[topics_mask]]
        if names_dict is not None:
            #подберем для тематик названия
            true_namings = create_namings(x.columns, names_dict[category][sentiment])
            x = rename_topics(x, true_namings)

        #создадим бинарные столбцы, необходимые для подсчета вовлеченности бренда в тематику
        #x_int = ((x.T)/x.sum(axis=1)).T    
        alpha = 0.3
        x_int = (x>alpha).astype(int)

        #но оставим вероятности принадлежности к тематикам для вывода наиболее вероятных кандидатов
        x.columns = ['topic_prob_'+el for el in x.columns]
        x_int.columns = ['topic_int_'+el for el in x_int.columns]

        #присоединим все к таблице
        df_sent = df_sent.reset_index(drop=True)
        df_sent = df_sent.join(x_int)
        df_sent = df_sent.join(x)
        
        df_sent = df_sent.loc[:, ~df_sent.columns.duplicated()]
        
        final_df.append(df_sent)
    final_df = pd.concat(final_df, axis=0)
    
    return final_df


"""Функции для построения таксономического описания пользователей"""

global_token = 'ef17c2b5ef17c2b5ef17c2b5d6ef72142eeef17ef17c2b5b494134878d23e70460f49a8'
global_version = 5.92

def vk_users_upload(id_users):
    """возвращает сырую информацию о пользователях по id или короткому имени"""
    token = global_token
    version = global_version
    user_ids = id_users
    fields = 'sex, bdate, city, country, relation, education, faculty_name'
    
    respons = requests.get('https://api.vk.com/method/users.get?', 
                           params = {
                                'access_token': token,
                                'v' : version,
                                'fields':fields,
                                'user_ids': user_ids,
                                'lang':'ru'})
    data = respons.json()['response']
    return data

def vk_get_users_info(all_users):
    """возвращает обработанную информацию о пользователях"""
    all_users = [str(x) for x in all_users]
    total = len(all_users) # количество id для выгрузки 
    #мы можем выгружать id по ~200шт.
    start = 0
    finish = 200
    #массив для результата
    answer = []
    
    while start<total:
        df = vk_users_upload(', '.join(all_users[start:finish]))
        start += 200
        finish += 200
        time.sleep(0.5)
        len(df)
        answer.extend(df)
        
    answer = pd.DataFrame(answer)
    
    answer['relation'] = answer['relation'].apply(relation)

    return answer

def relation(x):
    """возращает статус личных взаимоотношений"""
    relation_dict = {    
        1 : 'не женат/не замужем',
        2 : 'есть друг/есть подруга',
        3 : 'помолвлен/помолвлена',
        4 : 'женат/замужем',
        5 : 'всё сложно',
        6 : 'в активном поиске',
        7 : 'влюблён/влюблена',
        8 : 'в гражданском браке',
        0 : 'не указано'
    }    
    try:        
        ans = relation_dict[x]
    except:
        ans = ""
    return ans


def date_to_age(bdate):
    bdate = str(bdate)
    if len(bdate.split('.'))==3:
        d, m, y = bdate.split('.')
        d = int(d)
        m = int(m)
        y = int(y)
        return int((date.today()-date(day=d, month=m, year=y)).days/365.25)
    else:
        return '-'
    

def vk_get_subscr_batch(id_users, offset=0, count=100):
    """возвращает count подписок пользователей, начиная с offset"""
    token = global_token
    version = global_version
    metod = 'users.getSubscriptions'
    #в поле fields не должно быть запятой.... иначе оно не работает. 
    #поля должны быть разделены ТОЛЬКО запятой, без пробела
    response = requests.get(f'https://api.vk.com/method/{metod}?', 
                           params = {
                                'access_token': token,
                                'v' : version,
                                'user_id': id_users,
                                'extended': 1,
                                'offset': offset,
                                'count': count,
                                'fields': 'description,members_count,activity'})
    data = []
    if 'error' not in response.json():
        for el in response.json()['response']['items']:
            if ('id' in el)&('description' in el)&('members_count' in el):
                data.append({
                        'id': el['id'], 
                        'text': el['name'] + ' ' + el['description'],
                        'members_count': el['members_count'],
                        'activity': el['activity']
                        })
    return data

def upload_new_ids(ids, basic_info=False, groups_info=False):
    """Подгружает информацию о недостающих айдишниках"""
    #подругаем общую информацию профиля
    ids = [str(int(float(i))) for i in ids]
    if isinstance(basic_info, pd.DataFrame):
        basic_info = basic_info.rename(columns={'id': 'vk_id'}).astype(str)
        uploaded_ids = basic_info['vk_id'].unique()
        needed_ids = [i for i in ids if i not in uploaded_ids]
    else:
        needed_ids = ids
        basic_info = pd.DataFrame()

    if len(needed_ids)>0:
        print('Загрузка информации о пользователях')
        uploaded_info = vk_get_users_info(ids)
        uploaded_info = uploaded_info[['id', 'city', 'country', 'bdate', 'faculty_name', 'relation']].rename(columns={'id': 'vk_id'}).dropna()
        uploaded_info['city'] = uploaded_info['city'].fillna('-').apply(lambda x: x['title'] if x!='-' else x)
        uploaded_info['country'] = uploaded_info['country'].fillna('-').apply(lambda x: x['title'] if x!='-' else x)
        uploaded_info['age'] = uploaded_info['bdate'].apply(date_to_age)

        basic_info = pd.concat([basic_info, uploaded_info], sort=False)
        basic_info = basic_info.reset_index(drop=True)
     
    #подгружаем описание подписок
    if isinstance(groups_info, pd.DataFrame):
        groups_info = groups_info.rename(columns={'user_id': 'vk_id'}).astype(str)
        uploaded_ids = groups_info['vk_id'].unique()
        needed_ids = [i for i in ids if i not in uploaded_ids]
    else:
        needed_ids = ids
        groups_info = pd.DataFrame()
        
    if len(needed_ids)>0:
        print('Загрузка информации о подписках пользователей')
        #создание словаря с подписками списка юзеров
        groups_dict = dict(zip(needed_ids, 
                        [vk_get_subscr_batch(i, count=100) for i in tqdm(needed_ids)]))    
        #преобразование словаря в dataFrame
        columns = ['vk_id', 'group_id', 'text', 'members_count', 'activity']
        table = []
        for user_id in groups_dict:
            if len(groups_dict[user_id])>0:
                for group in groups_dict[user_id]:
                    table.append([user_id, group['id'], group['text'], group['members_count'], group['activity']])
            else:
                table.append([user_id, '', '', ''])
        groups_info = pd.concat([groups_info, pd.DataFrame(table, columns=columns)], sort=False)
        groups_info = groups_info.reset_index(drop=True)
        
        basic_info['vk_id'] = basic_info['vk_id'].astype(int).astype(str)
        groups_info['vk_id'] = groups_info['vk_id'].astype(int).astype(str)
        
    return (basic_info, groups_info)

#Таксономия
taxonomy = pd.read_csv('../../DATA/Taxonomy/Custom_Taxonomy_Vk.csv', encoding='cp1251', sep=';', header=0, index_col=(0, 1))

    
def vk_id(x):
    x = str(x)
    if ('vk.com' not in x):
        return ''
    else:
        return re.sub('[a-zA-Z]*', '', x.split('/')[-1])

def f_groups(x, taxonomy=taxonomy):
    edu_dict = dict(taxonomy.loc[('Высшее образование', ), 'info_lemms'])
    for group in edu_dict.keys():
        lemms = re.sub(';$', '', edu_dict[group].strip()).split('; ')
        for w in lemms:
            if w in str(x).lower():
                return group
    return np.nan
    
def segment_search_fast(descr, taxonomy, th_count=2, th_frac=0.1):
    """Функция поиска подходящего сегмента, опирающаяся на описание группы"""
    max_count = 0
    nearest_topic = ('', '')
    taxonomy = taxonomy.dropna(subset=['group_descr_tokens'])
    if len(descr)>0:
        for p_seg, seg in taxonomy.index:
            count = 0
            topic = re.sub(';$', '', taxonomy.loc[(p_seg, seg), 'group_descr_tokens'].strip()).split('; ')
            #Просто считаем, сколько раз встречаются леммы из описания в токенах, описывающих сегмент
            for word in descr:            
                if word in topic:
                    count += 1
            if count>max_count:
                nearest_topic = [p_seg, seg]
                max_count = count                
        if (max_count>=th_count)&(max_count/len(descr)>=th_frac):            
            return nearest_topic
        else:
            return ('', '')
    else:
        return ('', '')
    
def broad_segment_search(row, taxonomy):
    """функция поиска подходящего сегмента, позволяющая использовать встроенную описательную характеристику Вконтакте activity."""
    x = row['text_lemm']
    tag = row['activity']
    x = list(set(x.split()))
    ttag = taxonomy['vk_activity'].fillna('').apply(lambda x: tag in x).to_list()
    #Если мы нашли строки с таким activity
    if sum(ttag)>0:
        #Если ровно один, то проверим, есть ли хоть один токен в названии
        if sum(ttag)==1:
            p_seg, seg = taxonomy.loc[ttag, :].index[0]
            topic = re.sub(';$', '', taxonomy.loc[(p_seg, seg), 'group_descr_tokens'].strip()).split('; ')
            tokens_count = 0
            for token in topic:
                if token in x:
                    tokens_count =+ 1
            if tokens_count==0:
                p_seg, seg = segment_search_fast(x, taxonomy)
        #А если больше, то поищем среди подходящих
        else:
            topic = reduce((lambda a, b: a+b),  [re.sub('; *$', '', topic_tokens).strip().split('; ') for topic_tokens in taxonomy.loc[ttag, 'group_descr_tokens'].dropna().values])
            tokens_count = 0
            for token in topic:
                if token in x:
                    tokens_count =+ 1
            if tokens_count==0:
                p_seg, seg = segment_search_fast(x, taxonomy)
            else:
                p_seg, seg = segment_search_fast(x, taxonomy.loc[ttag, :])
    #Если же нет ни одного, то будем по старинке искать по описанию.
    else:
        p_seg, seg = segment_search_fast(x, taxonomy)
    return (p_seg, seg)

def taxonomy_concat(vk_dfs, df_to_concat_to=None):
    """
    Вычисляет принадлежность пользователей к сегментам таксономии и возвращает бинарную таблицу.
    vk_df: pd.DataFrame, таблица с описаниями групп из вк
    total_df: pd.DataFrame, таблица, к которой нужно присоединить итоговые сегменты
    """
    users_df, groups_df = vk_dfs
  
    users_df = users_df.rename(columns={'id': 'vk_id'})
    groups_df = groups_df.rename(columns={'user_id': 'vk_id'})
    users_df['vk_id'] = users_df['vk_id'].astype(int).astype(str)
    groups_df['vk_id'] = groups_df['vk_id'].astype(int).astype(str)
    users_df = users_df.reset_index(drop=True)
    groups_df = groups_df.reset_index(drop=True)
    
    #если общая таблица не задана, то надо создать ее
    if df_to_concat_to is None:
        df_to_concat_to = pd.DataFrame({'vk_id': list(users_df['vk_id'].unique())})
        df_to_concat_to = df_to_concat_to.drop_duplicates()
    else:
        users_df = users_df[users_df['vk_id'].isin(list(df_to_concat_to['vk_id'].unique()))]
        groups_df = groups_df[groups_df['vk_id'].isin(list(df_to_concat_to['vk_id'].unique()))]
    
    groups_df['activity'] = groups_df['activity'].fillna('')
    
    #лемматизация описания
    groups_df['text_lemm'] = lemm_text(groups_df['text'], russian_stopwords+vk_stopwords)
    #поиск сегментов
    tqdm.pandas()
    unique_groups_df = groups_df.drop_duplicates(subset=['group_id'])
    unique_groups_df.index = unique_groups_df['group_id']
    unique_groups_df = pd.DataFrame(list(unique_groups_df[['text_lemm', 'activity']].progress_apply(
            lambda x: broad_segment_search(x, taxonomy), axis=1)), columns=[
                    'p_segment', 'segment'], index=unique_groups_df.index)
    unique_groups_df = unique_groups_df.reset_index()
    groups_df = groups_df.merge(unique_groups_df, how='left', on='group_id')
    
    
        
    #подготавливаем пустые столбцы с названиями сегментов
    segments = taxonomy.index
    segment_columns = []
    
    for s1, s2 in segments:
        col_name = f'({s1}, {s2})'
        segment_columns.append(col_name)
        df_to_concat_to[col_name] = 0

    #Это бинаризация.
    #Ее можно было бы сделать встренными методами Pandas, 
    #но нам, во-первых, все равно придется проверять, что все столбцы создались
    #а еще образование все равно так построить не получится.
    #к тому же основные времязатраты у нас выше
    for i in groups_df.index:
        user_id = groups_df.loc[i, :]['vk_id']
        p_segment = groups_df.loc[i, :]['p_segment']
        segment = groups_df.loc[i, :]['segment']
        if (p_segment!='')&(segment!=''): 
            df_to_concat_to.loc[(df_to_concat_to['vk_id']==user_id), f'({p_segment}, {segment})'] = 1
            
    #Поищем образование по ключевым леммам        
    users_df['edu'] = users_df['faculty_name'].apply(f_groups).fillna('')
    for i in users_df.index:
        user_id = users_df.loc[i, :]['vk_id']
        p_segment = 'Высшее образование'
        segment = users_df.loc[i, :]['edu']
        if (p_segment!='')&(segment!=''):
            df_to_concat_to.at[(df_to_concat_to['vk_id']==user_id), f'({p_segment}, {segment})'] = 1
    
    return df_to_concat_to


def power_calculating(df, categories, theme_stopwords, theme_brands, params):
    """
    Средняя суммарная вероятность принадлежности к тематике 
    при моделировании с помощью ансамля NMF уменьшается с увеличением размеры выборки.
    Причем зависимочть эта степенная.
    С помощью данной функции можно подобрать такую степень, чтобы при возведении в нее 
    каждой вероятности в сумме получить почти единицу.
    
    Все параметры, как в build_topics
    """
    parts = {}
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        part_df = df.sample(frac=frac)
        total_df = []
        for category in tqdm(categories):
            cat_df = build_topics(part_df, category, (theme_stopwords, theme_brands), \
                                      params=params[category])

            cat_df['category'] = category
            total_df.append(cat_df)
        parts[frac] = pd.concat(total_df, sort=False)
    frac_list = {}
    keys = sorted(list(parts.keys()))
    for frac in keys:
        df = parts[frac].copy()
        tcols = [col for col in df.columns if '_prob' in col]
        frac_list[math.log(frac)] = math.log(df[tcols].sum(axis=1).mean())
    coord = pd.DataFrame(list(frac_list.values()), index=frac_list.keys())
    power = (coord[0].max()-coord[0].min())/(coord.index.max()-coord.index.min())
    return power

def create_tax_clusters(df_, cl_num=6, taxonomy=taxonomy, aff_type='global'):  
    """функция позволяет разбить аудиторию вк на кластеры по аудиторным признакам"""
    df = df_.copy()
    deleted_index = df[df['source']!='vk.com'].index.to_series().to_list()
    df = df[df['source']=='vk.com']
    
    mean_tax_values = taxonomy['mean_value'].copy()
    mean_tax_values.index = mean_tax_values.index.to_series().apply(lambda x: f'({x[0]}, {x[1]})').values
    taxonomy_columns = mean_tax_values[mean_tax_values>0.005].index.to_series()
    
#     taxonomy_columns = [col for col in df.columns if ('('==col[0])and(')'==col[-1])]
    taxonomy_columns = [col for col in taxonomy_columns if 'Высшее образование' not in col]
    topic_columns = [col for col in df.columns if '_prob' in col]
    deleted_index = deleted_index + df[df[taxonomy_columns].sum(axis=1)<=2].index.to_series().to_list()
    df = df[df[taxonomy_columns].sum(axis=1)>2]    
    
    df_taxonomy = df[taxonomy_columns][df[taxonomy_columns].notna().sum(axis=1)>0]
    
    if aff_type=='global':
        df_taxonomy = df_taxonomy/mean_tax_values[list(taxonomy_columns)]
    elif aff_type=='local':
        df_taxonomy = df_taxonomy/df_taxonomy.mean()
    n = len(topic_columns)
    
    df_taxonomy = df_taxonomy-df_taxonomy.min()
    
    cl = SpectralClustering(cl_num, affinity = 'cosine', gamma=1)
    df_taxonomy['clusters'] = pd.Series(cl.fit_predict(df_taxonomy), index=df_taxonomy.index)
    df = pd.concat([df, df_taxonomy[['clusters']]], axis=1)

    return pd.concat([df, df_.loc[deleted_index]], sort=False)

def rename_clusters(df, segments_dict, taxonomy=taxonomy, aff_type='global'):
    """функция, ищущая подходящие названия длякластеров из заранее выбранных"""
    mean_tax_values = taxonomy['mean_value'].copy()
    mean_tax_values.index = mean_tax_values.index.to_series().apply(lambda x: f'({x[0]}, {x[1]})').values
    taxonomy_columns = mean_tax_values[mean_tax_values>0.005].index.to_series()
    taxonomy_columns = [col for col in taxonomy_columns if 'Высшее образование' not in col]
    
    df_taxonomy = df[taxonomy_columns+['clusters']][df[taxonomy_columns].notna().sum(axis=1)>0]
    
    if aff_type=='global':
        df_taxonomy[taxonomy_columns] = df_taxonomy[taxonomy_columns]/mean_tax_values[list(taxonomy_columns)]
    elif aff_type=='local':
        df_taxonomy[taxonomy_columns] = df_taxonomy[taxonomy_columns]/df_taxonomy[taxonomy_columns].mean()
        
    clusters = df_taxonomy.groupby('clusters').mean()
    rename_dict = {}
    for i in clusters.index:
        cluster_segments = list(clusters.loc[i, :].sort_values(ascending=False).index[:10])
        inter_max = 0
        cl_name = ''
        for cl in segments_dict:
            inter = len(set(segments_dict[cl]).intersection(set(cluster_segments)))
            if inter>inter_max:
                inter_max = inter
                if inter_max>len(segments_dict[cl])//4:
                    cl_name = cl
        rename_dict[i] = cl_name

    df['clusters'] = df['clusters'].replace(rename_dict)
    return df

def segment_choice(df, th, taxonomy=taxonomy):  
    """функция, разбивающая тематики на группы с походим аудиторным описанием"""
    df_seg = df[df['source']=='vk.com']
    taxonomy_columns = [col for col in df.columns if ('('==col[0])and(')'==col[-1])]
    taxonomy_columns = [col for col in taxonomy_columns if 'Высшее образование' not in col]
    topic_columns = [col for col in df.columns if '_prob' in col]
    n = len(topic_columns)
    
    nmf = NMF(3)
    topic_tax_df = pd.concat([df_seg[df_seg[col]>th][taxonomy_columns].mean()-df_seg[taxonomy_columns].mean() for col in topic_columns], axis=1)
    topic_tax_df.columns = topic_columns
    topic_tax_df = topic_tax_df.T.fillna(0)
    nmf_x =  pd.DataFrame(nmf.fit_transform(topic_tax_df-topic_tax_df.min()), index=topic_columns).T
    nmf_x = (nmf_x / nmf_x.sum())
    
    clusters = nmf_x.idxmax()
    for i in range(3):
        df_seg[f'group_of_topics_{i}'] = 0
        
    for topic in clusters.index:
        df_seg.loc[df_seg[topic]>0.2, f'group_of_topics_{clusters[topic]}'] = 1
        

    return pd.concat([df_seg, df[df['source']!='vk.com']])