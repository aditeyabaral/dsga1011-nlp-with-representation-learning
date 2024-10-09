import matplotlib.pyplot as plt


def load_beam_search_scores():
    scores = list()
    for i in range(1, 6):
        # iterate over the 5 output files
        with open(f"out_beam_{i}.txt") as f:
            line = f.readlines()[-1]
            # load the BLEU score
            score = float(line.split()[-1])
            scores.append(score)
    return scores


def plot_scores():
    scores = load_beam_search_scores()
    plt.plot(range(1, 6), scores, marker="o")
    plt.xlabel("Beam Size")
    plt.ylabel("BLEU Score")
    plt.title("Beam Size vs BLEU Score")
    plt.grid()
    plt.savefig("beam_search_scores.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_scores()
