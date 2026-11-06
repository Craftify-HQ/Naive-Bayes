import csv
import os
import random
import time
import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FONT = '/Library/Fonts/Arial Unicode.ttf'

if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
else:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Tahoma', 'Helvetica']

plt.rcParams['axes.unicode_minus'] = False
random.seed(42)
np.random.seed(42)

CSV = 'data/w_review_train.csv'
URL = 'https://raw.githubusercontent.com/wongnai/wongnai-corpus/master/review/review_dataset.zip'

if not os.path.exists(CSV):
    os.makedirs('data', exist_ok=True)
    urllib.request.urlretrieve(URL, 'data/review_dataset.zip')
    with zipfile.ZipFile('data/review_dataset.zip') as z:
        z.extract('w_review_train.csv', 'data')
    os.remove('data/review_dataset.zip')

csv.field_size_limit(10 ** 9)
reviews = []
stars = []
with open(CSV, encoding='utf-8', newline='') as f:
    for text, star in csv.reader(f, delimiter=';'):
        reviews.append(text)
        stars.append(int(star))
stars = np.array(stars)

stop_words = set(thai_stopwords())
stop_words.update(['ร้าน', 'กิน', 'อร่อย', 'นะ', 'คะ', 'ค่ะ', 'ก็', 'และ', 'ใน'])


def tokenize_text(text):
    tokens = []
    for w in word_tokenize(text, engine='newmm'):
        w = w.strip().lower()
        if w and w not in stop_words and len(w) > 1 and any(c.isalpha() for c in w):
            tokens.append(w)
    return tokens


start_time = time.time()
docs_tokens = [tokenize_text(r) for r in reviews]

N = len(reviews)
indices = list(range(N))
random.shuffle(indices)
split_point = int(N * 0.8)

train_idx = indices[:split_point]
test_idx = indices[split_point:]

vocab_dict = {}
for idx in train_idx:
    for token in docs_tokens[idx]:
        if token not in vocab_dict:
            vocab_dict[token] = len(vocab_dict)

vocab_size = len(vocab_dict)
vocabulary_words = np.array(list(vocab_dict.keys()))

TF_matrix = np.zeros((5, vocab_size))
class_doc_counts = np.zeros(5)

for idx in train_idx:
    star_class = stars[idx] - 1
    class_doc_counts[star_class] += 1
    for token in docs_tokens[idx]:
        if token in vocab_dict:
            TF_matrix[star_class, vocab_dict[token]] += 1

class_priors = class_doc_counts / class_doc_counts.sum()
total_words_per_class = TF_matrix.sum(axis=1, keepdims=True)

smoothed_probabilities = (TF_matrix + 1) / (total_words_per_class + vocab_size)

log_class_priors = np.log(class_priors)
log_smoothed_probabilities = np.log(smoothed_probabilities)


def predict_rating(token_list):
    valid_word_ids = [vocab_dict[w] for w in token_list if w in vocab_dict]
    scores = log_class_priors + log_smoothed_probabilities[:, valid_word_ids].sum(axis=1)
    return np.argmax(scores) + 1


y_true_test = stars[test_idx]
y_pred_test = np.array([predict_rating(docs_tokens[i]) for i in test_idx])
accuracy = np.mean(y_pred_test == y_true_test)

print("\n" + "=" * 40)
print("     SUMMARY PERFORMANCE & STATISTICS")
print("=" * 40)
print(f"จำนวนรีวิวทั้งหมด : {N} รีวิว")
print(f"สัดส่วน Train / Test : {len(train_idx)} / {len(test_idx)} รีวิว (80/20)")
print(f"ขนาด Vocabulary (V)  : {vocab_size} คำ")
print(f"ความแม่นยำ (Accuracy) : {round(accuracy * 100, 2)}%")
print("-" * 40)
print("จำนวนคำรีวิวในแต่ละเรตติ้ง (Train Set):")
for s in range(5):
    print(
        f"   - เรตติ้ง {s + 1} ดาว : จำนวน {int(class_doc_counts[s])} รีวิว | คำทั้งหมด {int(total_words_per_class[s, 0])} คำ")
print("=" * 40 + "\n")

total_word_freq = TF_matrix.sum(axis=0)
MIN_WORD_FREQ = 300


def compute_unique_distinct_score(class_idx):
    other_classes_words_sum = total_word_freq - TF_matrix[class_idx]
    other_total_count = total_words_per_class.sum() - total_words_per_class[class_idx, 0]
    other_prob = (other_classes_words_sum + 1) / (other_total_count + vocab_size)

    distinct_score_val = np.log(smoothed_probabilities[class_idx] / other_prob)
    distinct_score_val[total_word_freq < MIN_WORD_FREQ] = -9999
    return distinct_score_val


candidate_words = []
for s in range(5):
    scores_s = compute_unique_distinct_score(s)
    top_indices = np.argsort(-scores_s)
    count = 0
    for idx_val in top_indices:
        word_str = vocabulary_words[idx_val]
        is_thai_only = all('\u0E00' <= c <= '\u0E7F' for c in word_str)
        if scores_s[idx_val] > -9000 and is_thai_only:
            candidate_words.append((word_str, idx_val, scores_s[idx_val]))
            count += 1
            if count >= 6:
                break

candidate_words.sort(key=lambda x: x[2], reverse=True)
selected_word_indices = []
for word_str, idx_val, score_val in candidate_words:
    if idx_val not in selected_word_indices:
        selected_word_indices.append(idx_val)

heatmap_matrix = smoothed_probabilities[:, selected_word_indices] * 1000

plt.figure(figsize=(max(14, len(selected_word_indices) * 0.45), 4.5))
sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt='.1f',
    cmap='YlGnBu',
    cbar_kws={'label': 'P(Word | Star) x 1000'},
    xticklabels=vocabulary_words[selected_word_indices],
    yticklabels=[f'{s} ดาว' for s in range(1, 6)]
)
plt.title('Top Distinctive Words Heatmap per Star Rating (Sorted)', fontsize=13, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


def generate_wordcloud_image(frequencies_dict):
    return WordCloud(
        font_path=FONT, width=600, height=400,
        background_color='white', max_words=50, random_state=42
    ).generate_from_frequencies(frequencies_dict)


cloud_datasets = []
top_overall_indices = np.argsort(-total_word_freq)[:80]
sorted_overall_freq = {vocabulary_words[i]: float(total_word_freq[i]) for i in top_overall_indices}
cloud_datasets.append(('ภาพรวมทุกดาวทั้งหมด', sorted_overall_freq))

for s in range(5):
    scores_s = compute_unique_distinct_score(s)
    top_class_indices = np.argsort(-scores_s)
    sorted_class_freq = {}
    for i in top_class_indices:
        if scores_s[i] > 0:
            sorted_class_freq[vocabulary_words[i]] = float(scores_s[i])
            if len(sorted_class_freq) >= 50:
                break
    cloud_datasets.append((f'คำเด่นเฉพาะเรตติ้ง {s + 1} ดาว', sorted_class_freq))

fig_wc, sub_axes = plt.subplots(2, 3, figsize=(16, 9))
for target_ax, (title_text, freq_data) in zip(sub_axes.flatten(), cloud_datasets):
    target_ax.imshow(generate_wordcloud_image(freq_data))
    target_ax.set_title(title_text, fontsize=13, fontweight='bold')
    target_ax.axis('off')

plt.tight_layout()
plt.show()