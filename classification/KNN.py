import random
import statistics
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from pathlib import Path


def distance(point_a, point_b):
    return (
        ((point_a["aspect_ratio"] - point_b["aspect_ratio"]) ** 2) +
        ((point_a["corners"] - point_b["corners"]) ** 2)
    ) ** 0.5  # Euclidean


class KNN():
    max_iterations = 100

    def __init__(self, training_percentage: float, data: str, k: int = 4, model: str = None):
        self.k = k
        if model:
            df = pd.read_csv(model)
            self.centroids = df.to_dict('records')
            random.seed()
            df = pd.read_csv(data, index_col=0)
            self.dataset: List[Dict] = df.to_dict('records')
            random.shuffle(self.dataset)
            index = int(len(self.dataset) * training_percentage)
            self.test_set = self.dataset[index:]
        else:
            random.seed()
            df = pd.read_csv(data, index_col=0)
            self.dataset: List[Dict] = df.to_dict('records')
            random.shuffle(self.dataset)
            index = int(len(self.dataset) * training_percentage)
            self.train_set = self.dataset[:index]
            self.test_set = self.dataset[index:]
            self.centroids = self.train()
            df = pd.DataFrame(self.centroids)
            df.to_csv("models/knn.csv")

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
        pass


if __name__ == "__main__":
    knn = KNN(0.35, "data/dataset.csv")
    accuracy = knn.test()
    print("Model Accuracy = ", accuracy)
