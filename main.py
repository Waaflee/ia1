import pandas as pd
from matplotlib import pyplot as plt
from setup import setup


data = "data/dataset.csv"


def main() -> None:
    df = pd.read_csv(data, index_col=0)
    dataset: List[Dict] = df.to_dict('records')
    print(dataset)

    # # Plot the data
    # plt.figure(figsize=(6, 6))
    # plt.scatter([i["aspect_ratio"] for i in dataset],
    #             [i["corners"] for i in dataset])
    # plt.xlabel('Aspect Ratio')
    # plt.ylabel('Amount of Corners')
    # plt.title('Visualization of Extracted Features')
    # plt.show()


if __name__ == "__main__":
    setup()
    main()
