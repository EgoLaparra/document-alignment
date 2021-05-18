import csv
import joblib
import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def load_titles(csv_path, title_column):
    document_list = []
    with open(csv_path, encoding="utf8") as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        next(reader)
        for document in reader:
            title = document[title_column]
            document_list.append(title)
    return document_list


def process_efficient_mode(source, target, clf, vectorizer):
    vectors = vectorizer.fit_transform(source + target)
    source_vectors = vectors[:len(source)]
    target_vectors = vectors[len(source):]
    similarities = cosine_similarity(source_vectors, target_vectors)
    predictions = clf.predict(similarities.reshape(-1, 1))
    predictions = predictions.reshape(len(source), -1)
    sorted_by_sim = np.argsort((-similarities), axis=1)
    alignments = {}
    for source_id, source_title in enumerate(source):
        alignments[source_id] = []
        for target_id in sorted_by_sim[source_id]:
            similarity = similarities[source_id][target_id]
            prediction = predictions[source_id][target_id]
            alignment = {"id": int(target_id),
                         "sim": similarity,
                         "prediction": bool(prediction)}
            alignments[source_id].append(alignment)
    return alignments


def process_original_mode(source, target, clf, vectorizer):
    alignments = {}
    for source_id, source_title in enumerate(tqdm(source)):
        alignments[source_id] = []
        for target_id, target_title in enumerate(tqdm(target, leave=False)):
            title_vector = vectorizer.fit_transform([source_title, target_title])
            [[_, one_pair_cos_sim], [_, _]] = cosine_similarity(title_vector)
            if one_pair_cos_sim > 0.:
                cos_sim = np.array(one_pair_cos_sim).reshape(-1, 1)
                prediction = clf.predict(cos_sim)
                alignment = {"id": target_id,
                             "sim": one_pair_cos_sim,
                             "prediction": bool(prediction)}
                alignments[source_id].append(alignment)
        alignments[source_id].sort(key=lambda x: x["sim"], reverse=True)
    return alignments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align documents based on title.')
    parser.add_argument('-s', '--source', dest="source", metavar="FILE",
                        help='Source input CSV file.')
    parser.add_argument('-t', '--target', dest="target", metavar="FILE",
                        help='Target input CSV file.')
    parser.add_argument('-o', '--output', dest="output", metavar="FILE",
                        help='Output JSON file.')
    parser.add_argument('-m', '--model', dest="model", metavar="FILE",
                        help='Model JOBLIB file.', default="model.joblib")
    parser.add_argument('-v', '--vectorizer', dest="vectorizer", metavar="FILE",
                        help='Vectorizer JOBLIB file.', default="vectorizer.joblib")
    parser.add_argument('-e', '--efficient', dest="efficient", action="store_true",
                        help='Run efficient mode instead of original mode.')
    args = parser.parse_args()

    clf = joblib.load(args.model)
    vectorizer = joblib.load(args.vectorizer)
    source = load_titles(args.source, 1)
    target = load_titles(args.target, 0)
    if args.efficient:
        alignments = process_efficient_mode(source, target, clf, vectorizer)
    else:
        alignments = process_original_mode(source, target, clf, vectorizer)
    with open(args.output, "w", encoding="utf8") as json_file:
        json.dump(alignments, json_file)
