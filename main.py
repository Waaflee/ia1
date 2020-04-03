import pandas as pd
from matplotlib import pyplot as plt
from classification.KNN import KNN
from classification.KMeans import KMeans


data = "data/dataset.csv"


def main() -> None:
    print("-"*25)
    print("KNN: ")
    knn = KNN(0.45, "data/dataset.csv")
    accuracy = knn.test()
    print("Model Accuracy = ", accuracy)
    print("KMeans: ")
    km = KMeans(0.45, "data/dataset.csv")
    accuracy = km.test()
    print("Model Accuracy = ", accuracy)
    print("-"*25)
    # df = pd.read_csv(data, index_col=0)
    # dataset: List[Dict] = df.to_dict('records')
    # print(dataset)

    # # Plot the data
    # plt.figure(figsize=(6, 6))
    # plt.scatter([i["aspect_ratio"] for i in dataset],
    #             [i["corners"] for i in dataset])
    # plt.xlabel('Aspect Ratio')
    # plt.ylabel('Amount of Corners')
    # plt.title('Visualization of Extracted Features')
    # plt.show()


if __name__ == "__main__":
    main()
