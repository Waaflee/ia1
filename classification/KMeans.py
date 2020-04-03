import random
import statistics
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from pathlib import Path


def distance(point_a, point_b):
    return np.linalg.norm(point_a["array"]-point_b["array"])


class KMeans():
    max_iterations = 100

    def __init__(self, training_percentage: float, data: str, k: int = 4, model: str = None):
        # Load requested model form disk if exists
        if model and Path(model).is_file():
            df = pd.read_csv(model)
            self.centroids = df.to_dict('records')
            random.seed()
            df = pd.read_csv(data, index_col=0)
            self.dataset: List[Dict] = df.to_dict('records')
            random.shuffle(self.dataset)
            index = int(len(self.dataset) * training_percentage)
            self.test_set = self.dataset[index:]
        # Train model and save it to disk
        else:
            random.seed()
            self.k = k
            df = pd.read_csv(data, index_col=0)
            # self.dataset: List[Dict] = df.to_dict('records')
            self.dataset: List[Union[float, str]] = df.to_numpy()
            dataset = []
            for index, i in enumerate(self.dataset):
                num_data = i[:-1]
                label = i[-1]
                new_data = {"array": num_data, "label": label}
                dataset.append(new_data)
            self.dataset = dataset

            random.shuffle(self.dataset)
            index = int(len(self.dataset) * training_percentage)
            self.train_set = self.dataset[:index]
            self.test_set = self.dataset[index:]
            self.centroids = self.train()
            df = pd.DataFrame(self.centroids)
            df.to_csv("models/kmeans.csv")

    def train(self) -> List[Union[int, float, str]]:
        centroids: List = random.sample(list(self.train_set), self.k)
        for index, i in enumerate(centroids):
            i["centroid"] = index
        acc = 0
        while acc < self.max_iterations:
            acc += 1
            new_centroids = []
            # assign point to centroid
            for index, point in enumerate(self.train_set):
                distaces: List[float] = []
                for centroid in centroids:
                    distaces.append(distance(point, centroid))
                closest_centroid = centroids[distaces.index(min(distaces))]
                assigned_point = point.copy()
                assigned_point["centroid"] = closest_centroid["centroid"]
                self.train_set[index] = assigned_point
            # recalculate centroids
            for centroid in centroids:
                points = [i for i in self.train_set if i["centroid"]
                          == centroid["centroid"]]
                mean = np.mean([i["array"] for i in points], axis=0)
                point = min(points, key=lambda x: abs(
                    np.linalg.norm(x["array"]-mean)))
                new_centroids.append(point)

            if np.array_equal(np.array(list(centroids)), np.array(list(new_centroids))):
                print(acc)
                return centroids
            else:
                centroids = new_centroids
        else:
            return centroids

    def test(self) -> float:
        acc = 0
        for i in self.test_set:
            classification = self.classify(i)
            if classification == i["label"]:
                acc += 1
        return acc / len(self.test_set)

    def classify(self, data: Dict[str, Union[str, int, float]]) -> str:
        distaces: List[float] = []
        for centroid in self.centroids:
            distaces.append(distance(data, centroid))
        closest_centroid = self.centroids[distaces.index(min(distaces))]
        return closest_centroid["label"]


if __name__ == "__main__":
    km = KMeans(0.45, "data/dataset.csv")
    accuracy = km.test()
    print("Model Accuracy = ", accuracy)
