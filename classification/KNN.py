import random
from collections import Counter
from typing import Dict, List, Union
import pandas as pd
import numpy as np


def distance(point_a, point_b):
    return np.linalg.norm(point_a["array"]-point_b["array"])


class KNN():
    def __init__(self, training_percentage: float, data: str, k: int = 3, model: str = None):
        self.k = k
        random.seed()
        df = pd.read_csv(data, index_col=0)
        self.dataset: List[Union[float, str]] = df.to_numpy()
        dataset = []
        for index, i in enumerate(self.dataset):
            num_data = i[:-1]
            label = i[-1]
            new_data = {"array": num_data, "label": label}
            dataset.append(new_data)
        self.dataset = dataset
        # self.dataset: List[Dict] = df.to_dict('records')
        random.shuffle(self.dataset)
        index = int(len(self.dataset) * training_percentage)
        self.train_set = self.dataset[:index]
        self.test_set = self.dataset[index:]

    def train(self) -> List[Dict]:
        pass

    def test(self) -> float:
        acc = 0
        for i in self.test_set:
            classification = self.classify(i)
            if classification == i["label"]:
                acc += 1
        return acc / len(self.test_set)

    def classify(self, data: Dict[str, Union[str, int, float]]) -> str:
        dataset = self.train_set.copy()
        for i in dataset:
            i["distance"] = distance(data, i)
        dataset = sorted(dataset, key=lambda x: x["distance"])
        closest_k = dataset[:self.k]
        closest_k = [i["label"] for i in closest_k]
        occurrence_count = Counter(closest_k)
        return occurrence_count.most_common(1)[0][0]


if __name__ == "__main__":
    knn = KNN(0.45, "data/dataset.csv")
    accuracy = knn.test()
    print("Model Accuracy = ", accuracy)
