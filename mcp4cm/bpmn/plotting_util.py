from matplotlib import pyplot as plt


def plot_duplicate_piechart(labels: tuple, sizes: tuple, title: str) -> None:
    title = title if title is not None else "Proportions"
    if len(labels) != len(sizes):
        raise ValueError("Number of labels and sizes do not match")

    colors = ['green', 'red']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title(title)
    plt.show()


def plot_tf_idf_graphs(steps, near_duplicate_percentages, duplicat_groups):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].plot(steps, near_duplicate_percentages)
    axes[0].set_xlabel("Similarity Threshold")
    axes[0].set_ylabel("% Near Duplicates")


    axes[1].plot(steps, duplicat_groups)
    axes[1].set_xlabel("Similarity Threshold")
    axes[1].set_ylabel("Number of Duplicate Groups")

    plt.tight_layout()
    plt.show()
