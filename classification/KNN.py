import random
import statistics
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def distance(point_a, point_b):
    return (
        ((point_a["aspect_ratio"] - point_b["aspect_ratio"]) ** 2) +
        ((point_a["corners"] - point_b["corners"]) ** 2)
    ) ** 0.5  # Euclidean


class KNN():
    def __init__(self, training_percentage: float, data: str, k: int = 3, model: str = None):
        self.k = k
        random.seed()
        df = pd.read_csv(data, index_col=0)
        self.dataset: List[Dict] = df.to_dict('records')
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
            if classification == i["classification"]:
                acc += 1
        return acc / len(self.test_set)

    def classify(self, data: Dict[str, Union[str, int, float]]) -> str:
        dataset = self.train_set.copy()
        for i in dataset:
            i["distance"] = distance(data, i)
        dataset = sorted(dataset, key=lambda x: x["distance"])
        closest_k = dataset[:self.k]
        closest_k = [i["classification"] for i in closest_k]
        occurrence_count = Counter(closest_k)
        return occurrence_count.most_common(1)[0][0]


if __name__ == "__main__":
    knn = KNN(0.4, "data/dataset.csv")
    accuracy = knn.test()
    print("Model Accuracy = ", accuracy)
