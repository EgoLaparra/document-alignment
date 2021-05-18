import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from title_alignment_classifier import splitDataFrameIntoSmaller, process_data, train_model_and_test_model
import csv
from joblib import dump



if __name__ == '__main__':

    # train logistic regression classifier on title_alignment dataset

    df = pd.read_csv("Title_Alignment.csv", header=0)
    train_new_df = df.iloc[:, [3, 11]][0:7308]  # 1044
    test_new_df = df.iloc[:, [3, 11]][7308:8127]  # 117

    train_list_of_chunks = splitDataFrameIntoSmaller(train_new_df, chunkSize=7)
    train_title_vector, train_cos_sim, train_label = process_data(train_list_of_chunks)
    dump(train_title_vector, 'vectorizer.joblib')


    test_list_of_chunks = splitDataFrameIntoSmaller(test_new_df, chunkSize=7)
    test_cos_sim, test_label = process_data(test_list_of_chunks)[1:]

    clf,train_accuracy = train_model_and_test_model(train_cos_sim, train_label, test_cos_sim)
    dump(clf, 'model.joblib')

    # Apply trained classifier on eis document
    dataset = pd.read_csv('document_records_with_column_names.csv', sep=',', header=0)
    ids = dataset.iloc[:, 0][0:6]# 14,635
    titles = dataset.iloc[:, 8][0:6]


    with open('pair_title.csv', 'w') as file:
        pair_writer = csv.writer(file)
        for (id1, title1), (id2, title2) in combinations(zip(ids, titles), 2):
            title_vector = train_title_vector.fit_transform([title1, title2])
            [[_, one_pair_cos_sim], [_, _]] = cosine_similarity(title_vector)
            cos_sim = np.array(one_pair_cos_sim).reshape(-1, 1)
            prediction = clf.predict(cos_sim)
            pair_writer.writerow([id1, id2, one_pair_cos_sim, prediction])


















