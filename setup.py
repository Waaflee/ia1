import pandas as pd
from typing import Iterable, List
from utils.processing import extract_features
import os

prefix: str = "dataset/"
folders: Iterable[str] = ["nails", "screws", "washers", "nuts"]


def setup() -> None:
    dataset: List[Dict[str, Union[int, str, float]]] = []
    for folder in folders:
        images: List[str] = os.listdir(f"{prefix}{folder}")
        for image in images:
            classification: str = folder[:-1]
            data = extract_features(f"{prefix}{folder}/{image}")
            data["classification"] = classification
            dataset.append(data)
    else:
        df = pd.DataFrame(dataset)
        print(df)
        df.to_csv("dataset.csv")


if __name__ == "__main__":
    setup()
