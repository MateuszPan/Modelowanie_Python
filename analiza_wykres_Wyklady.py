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
    words_list = [word for word, _ in sorted_words]

    return ranks, frequencies, words_list

def plot_zipf(ranks, frequencies, words_list, title="Prawo Zipfa"):
    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, frequencies, marker='.', linestyle='none', color='blue', label="Dane")

    for i in range(min(10, len(words_list))):
        plt.annotate(words_list[i], (ranks[i], frequencies[i]), fontsize=10, ha='right', color='red')

    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)
    
    fit_range = min(100, len(log_ranks))  
    slope, intercept = np.polyfit(log_ranks[:fit_range], log_frequencies[:fit_range], 1)
    
    alpha = -slope

    fit_frequencies = np.exp(intercept) * ranks ** slope
    plt.loglog(ranks, fit_frequencies, 'r--', label=f"Regresja: α ≈ {alpha:.2f}")

    plt.title(title)
    plt.xlabel("Ranga słowa (log)")
    plt.ylabel("Częstotliwość występowania (log)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Wykładnik Zipfa (α): {alpha:.2f}")

file_path = r"D:\Mati\STUDIA\VI Semestr\Mdelowanie\lista_4\Zad_1\Wyklady.txt"

text = load_text(file_path)
words = process_text(text)

ranks, frequencies, words_list = zipf_law_analysis(words)

plot_zipf(ranks, frequencies, words_list, title="Prawo Zipfa - Wyklady.txt")
