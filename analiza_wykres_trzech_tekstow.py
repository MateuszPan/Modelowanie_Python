import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return text

def process_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return words

def zipf_law_analysis(words):
    word_counts = Counter(words)
    sorted_words = word_counts.most_common()
    
    ranks = np.arange(1, len(sorted_words) + 1)
    frequencies = [count for _, count in sorted_words]

    return ranks, frequencies

def plot_zipf_multiple(file_paths, labels, colors):
    plt.figure(figsize=(8, 6))

    for file_path, label, color in zip(file_paths, labels, colors):
        text = load_text(file_path)
        words = process_text(text)
        ranks, frequencies = zipf_law_analysis(words)

        plt.loglog(ranks, frequencies, marker='.', linestyle='none', color=color, label=f"{label}")

        log_ranks = np.log(ranks)
        log_frequencies = np.log(frequencies)
        fit_range = min(100, len(log_ranks))
        slope, intercept = np.polyfit(log_ranks[:fit_range], log_frequencies[:fit_range], 1)
        alpha = -slope

        fit_frequencies = np.exp(intercept) * ranks ** slope
        plt.loglog(ranks, fit_frequencies, linestyle='dashed', color=color, label=f"{label} (\u03b1 \u2248 {alpha:.2f})")

    plt.title("Prawo Zipfa - Porównanie trzech tekstów")
    plt.xlabel("Ranga słowa (log)")
    plt.ylabel("Częstotliwość występowania (log)")
    plt.legend()
    plt.grid(True)
    plt.show()

file_paths = [
    r"D:\Mati\STUDIA\VI Semestr\Mdelowanie\lista_4\Zad_2\1000_slowek_ang.txt",
    r"D:\Mati\STUDIA\VI Semestr\Mdelowanie\lista_4\Zad_2\instrukcja.txt",
    r"D:\Mati\STUDIA\VI Semestr\Mdelowanie\lista_4\Zad_2\O psie.txt"
]

labels = ["1000 słówek ang.", "Instrukcja", "O psie"]
colors = ["blue", "green", "red"]

plot_zipf_multiple(file_paths, labels, colors)
