import pandas as pd
from typing import Iterable, List
from utils.processing import extract_features
import os
import multiprocessing

prefix: str = "dataset/"
folders: Iterable[str] = ["nails", "screws", "washers", "nuts"]


def setup() -> None:
    dataset: List[Dict[str, Union[int, str, float]]] = []
    for folder in folders:
        images: List[str] = os.listdir(f"{prefix}{folder}")
        with multiprocessing.Pool(processes=4) as pool:
            classification: str = folder[:-1]
            data = pool.map(extract_features, [
                            f"{prefix}{folder}/{image}" for image in images])
            for i in data:
                i["classification"] = classification
            dataset += data
    else:
        df = pd.DataFrame(dataset)
        print(df)
        df.to_csv("data/dataset.csv")


if __name__ == "__main__":
    setup()
