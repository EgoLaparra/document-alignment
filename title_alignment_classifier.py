import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


def splitDataFrameIntoSmaller(df, chunkSize = 7):
    """
    split a dataset by chunks

    :param df: dataset
    :param chunkSize: 7 because each project with documents occupy 7 rows on dataset
    :return: a list of chunks
    """
    listOfDf = []
    numberChunks = len(df) // chunkSize
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf



def process_data (list_of_chunks):
    """
    To precess dataset using TF-idf vectorizer
    To calculate cosine similarity between project title and each document title
    To collect labels to a list
    :param list_of_chunks: each chunk include one project title and 5 related document title
    :return: train_x and train_y
    """

    all_cosim = [] # collect cosine similarity for each title pair to a list
    label_data = [] # collect label data(false or true) to a list

    for i in list_of_chunks:

        title = i.iloc[:, 0].tolist()[0:6]

        label = i.iloc[:,1].tolist()[1:6]

        project_title = title[0]

        two_title = []

        for x in range(1, 6):
            two_title.append([project_title, title[x]]) # pair each project title with its document title
        cos_similarity = []
        for y in two_title:
            # use Tfidf to process each title pair
            title_vect = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')
            title_vector = title_vect.fit_transform(y)
            cos_similarity.append(cosine_similarity(title_vector)[0][1])
        all_cosim.append(cos_similarity)
        label_data.append(label)

    flat_cos_sim = [item for sublist in all_cosim for item in sublist] # change a list of lists to a list

    flat_label = [item for sublist in label_data for item in sublist] # change a list of lists to a list

    return flat_cos_sim, flat_label


def train_model_and_test_model(train_cos_sim, train_label, test_cos_sim):
    """
    Build logistic regression to train model and test model
    :param train_cos_sim: train_X
    :param train_label: train_y
    :param test_cos_sim: test_X
    :return: train_accuracy, predicted label in numpy array, predicted label and test_accuracy
    """
    # change the shape of training data so that it can be fed into classifier
    train_X = np.array(train_cos_sim).reshape(-1, 1)
    train_y = train_label
    # build a logistic regression classifier
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(train_X, train_y)
    train_pred_y = clf.predict(train_X) # use logisitic regression classifier to predict train label
    train_accuracy = accuracy_score(train_y, train_pred_y) # measure the accuracy between predict label and train label

    test_X = np.array(test_cos_sim).reshape(-1, 1)
    pred_y = clf.predict(test_X) # predict the label of test data
    pred_label = pred_y.tolist()  # change numpy array to list
    # test_accuracy = accuracy_score(test_label, pred_label) # measure the accuracy between predict label and test label

    return train_accuracy, pred_label




