import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from title_alignment_classifier import splitDataFrameIntoSmaller, process_data, train_model_and_test_model
import csv
import random


def random_title_pairs(min_t, max_t,new_title_pairs, cos_sin, predicted_out):
    """
    to generate 100 pairs for each threshold
    :param min_t: the minimum threshold for cosine similarity
    :param max_t: the maximum threshold for cosine similarity
    :param new_title_pairs: title_pairs: all pairs of titles
    :param cos_sin: cosine similarity of each title pairs
    :param predicted_out: the predicted outcome of each title pairs
    :return: 100 random samples
    """
    result = list(zip(new_title_pairs, cos_sin, predicted_out))
    dic={}
    for i in result:
        dic[i[0]]={}
        dic[i[0]]["cosine"]= i[1]
        dic[i[0]]["predicted_outcome"]=i[2]
    title = []
    cs = []
    pred = []
    for i in result:
        if min_t <= dic[i[0]]["cosine"] < max_t:
            title.append(i[0])
            cs.append(dic[i[0]]["cosine"])
            pred.append(dic[i[0]]["predicted_outcome"])

    new_result = list(zip(title, cs, pred))
    random_sample = random.sample(new_result, 100)

    return random_sample


def build_csv_file_based_on_threshold(min_t, max_t, new_pairs, cos_sim, predicted_label):
    """
    To build a csv file based on different threshold. In this csv, one column for title pairs. title paris are separated
    by comma. one column for cosine similarity. The last column for predicted label

    :param min_t:the minimum threshold for cosine similarity
    :param max_t:the maximum threshold for cosine similarity
    :param new_pairs:all pairs of titles
    :param cos_sim: cosine similarity of each title pairs
    :param pred_label:the predicted outcome of each title pairs
    :return: a csv file
    """
    new_pairs = [tuple(i) for i in new_pairs]
    new_pairs = ['   ,   '.join(i) for i in new_pairs] # two titles are separated by comma

    results = random_title_pairs(min_t, max_t, new_pairs, cos_sim, predicted_label)
    with open('pair_title.csv', 'w') as file:
        pair_writer = csv.writer(file)
        for t, c, p in results:
            pair_writer.writerow([t, c, p])

if __name__ == '__main__':

    # train logistic regression classifier on title_alignment dataset

    df = pd.read_csv("Title_Alignment.csv", header=0)
    train_new_df = df.iloc[:, [3, 11]][0:7308]  # 1044
    test_new_df = df.iloc[:, [3, 11]][7308:8127]  # 117

    train_list_of_chunks = splitDataFrameIntoSmaller(train_new_df, chunkSize=7)
    train_cos_sim, train_label = process_data(train_list_of_chunks)
    test_list_of_chunks = splitDataFrameIntoSmaller(test_new_df, chunkSize=7)
    test_cos_sim, test_label = process_data(test_list_of_chunks)
    train_accuracy, pred_label = train_model_and_test_model(train_cos_sim, train_label, test_cos_sim)


    # Apply trained classifier on eis document
    dataset = pd.read_csv('document_records_with_column_names.csv', sep=',', header=0)
    ids = dataset.iloc[:, 0]  # 14,635
    titles = dataset.iloc[:, 8]
    pair_ids = list(combinations(ids, 2))  # pair any of two ids on file
    pair_titles = list(combinations(titles, 2)) # pair any of two titles on file

    # print(len(pair_titles))

    cos_sim = []

    for pair in pair_titles:  # pair= [title 1 , title 2]

        title_vect = TfidfVectorizer()

        title_vector = title_vect.fit_transform(pair)

        cos_sim.append(cosine_similarity(title_vector)[0][1])

    # create a csv file that contains 3 columns: document 1 id, document 2 id, cosine similarity
    with open('pair_title.csv', 'w') as file:
        pair_writer = csv.writer(file)
        for ids, cs in zip(pair_ids, cos_sim):
            pair_writer.writerow([ids[0], ids[1], cs])

    # create a csv file that contains 3 columns: id: document title 1, id: document 2, cosine similarity
    pair_lst = []
    pair_2nd = []
    for i in zip(pair_ids, pair_titles):
        pair_lst.append([str(i[0][0]), i[1][0]])
        pair_2nd.append([str(i[0][1]), i[1][1]])

    pair_1 = [' : '.join(x) for x in pair_lst]
    pair_2 = [' : '.join(w) for w in pair_2nd]

    title_pair = list(zip(pair_1, pair_2))

    with open('pair_title.csv', 'w') as file:
        pair_writer = csv.writer(file)
        for title, cs in zip(title_pair, cos_sim):
            pair_writer.writerow([title[0], title[1], cs])


    # build_csv_file_based_on_threthold(0.6, 0.7, new_pairs, cos_sim, new_pred_y)










