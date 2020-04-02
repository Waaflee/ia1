import pandas as pd
from typing import Dict, Union, List
import random
import statistics


class KMeans():
    def __init__(self, training_percentage: float, data: str):
        random.seed()
        self.training = training_percentage
        df = pd.read_csv(data, index_col=0)
        self.dataset = df.to_dict('records')
        # self.nails: List[Dict] = df.loc[df['classification']
        #                                 == "nail"].to_dict('records')
        # self.screws: List[Dict] = df.loc[df['classification']
        #                                  == "screw"].to_dict('records')
        # self.washers: List[Dict] = df.loc[df['classification']
        #                                   == "washer"].to_dict('records')
        # self.nuts: List[Dict] = df.loc[df['classification']
        #                                == "nut"].to_dict('records')
        # self.nails = df.loc[df['classification'] == "nail"].values.tolist()
        # self.screws = df.loc[df['classification'] == "screw"].values.tolist()
        # self.washers = df.loc[df['classification'] == "washer"].values.tolist()
        # self.nuts = df.loc[df['classification'] == "nut"].values.tolist()

    def train(self):
        pass

    def test(self):
        pass

    def classify(self, data: Dict[str, Union[str, int, float]]) -> str:
        pass


if __name__ == "__main__":
    km = KMeans(0.35, "data/dataset.csv")
    km.train()
