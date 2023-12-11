import time
import csv
import matplotlib.pyplot as plt
import pandas as pd

from transformer import masked_transformer_api, standard_transformer_api


def measure_inference_time(text, model_api):
    start_time = time.time()
    result = model_api(text)
    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time, result


def write_to_csv(file_path, data):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def plot_data(file_path, save_path):
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Assuming CSV format: [Model Type, Text Length, Inference Time]
    model_types = [row[0] for row in data]
    text_lengths = [int(row[1]) for row in data]
    inference_times = [float(row[2]) for row in data]

    # Map model types to colors
    colors = ["blue" if model_type == "Masked" else "red" for model_type in model_types]

    plt.scatter(text_lengths, inference_times, c=colors)
    plt.xlabel("Text Length (num of characters)")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Time vs Text Length")

    # Add a legend
    plt.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Masked",
                markersize=10,
                markerfacecolor="blue",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Standard",
                markersize=10,
                markerfacecolor="red",
            ),
        ]
    )

    plt.savefig(save_path)


if __name__ == "__main__":
    csv_file_path = "inference_times_sms_data.csv"
    plot_file_path = "inference_times_plot_sms_data.png"
    dataset_path = "spam.csv"

    NUM_SAMPLES = 100

    df = pd.read_csv(dataset_path, encoding="ISO-8859-1")

    # Select 100 random sample texts
    random_samples = df["v2"].sample(n=NUM_SAMPLES, random_state=1)

    sample_texts = random_samples.tolist()
    masked_better_count = 0

    for text in sample_texts:
        standard_time, _ = measure_inference_time(text, standard_transformer_api)
        masked_time, _ = measure_inference_time(text, masked_transformer_api)

        write_to_csv(csv_file_path, ["Masked", len(text), masked_time])
        write_to_csv(csv_file_path, ["Standard", len(text), standard_time])

        if masked_time < standard_time:
            masked_better_count += 1

    print(
        f"\n\nMasked model was better {masked_better_count} times out of {len(sample_texts)}"
    )

    plot_data(csv_file_path, plot_file_path)
