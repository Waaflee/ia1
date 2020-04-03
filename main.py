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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    df = pd.read_csv(data, index_col=0)
    dataset: List[Dict] = df.to_dict('records')
    for i in dataset:
        ax.scatter(float(i['0']), float(i['1']), float(i['2']), marker='o')

    ax.set_xlabel('H[0]')
    ax.set_ylabel('H[1]')
    ax.set_zlabel('H[3]')
    plt.show()

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
