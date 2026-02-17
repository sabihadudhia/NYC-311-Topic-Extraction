import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print("="*80)
print("TEXT VECTORIZATION")
print("="*80)

file_path = r"C:\Users\sabih\OneDrive\Desktop\Project -  Data Analysis\preprocessed_data.csv"

try:
    df = pd.read_csv(file_path)
    corpus = df['Problem Detail (formerly Descriptor)'].fillna('').astype(str).values
    print(f"\nLoaded {len(corpus)} documents")
except FileNotFoundError:
    print("Error: preprocessed_data.csv not found.")
    exit()
except KeyError:
    print("Error: Column 'Problem Detail (formerly Descriptor)' not found.")
    exit()

print("\n" + "="*80)
print("TF-IDF VECTORIZATION")
print("="*80)

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"\nMatrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary sample: {list(feature_names[:10])}")

non_zero_counts = (tfidf_matrix != 0).sum(axis=1).A1
print(f"\nSummary:")
print(f"  Vocabulary size: {len(feature_names)}")
print(f"  Avg words/doc: {non_zero_counts.mean():.2f}")
print(f"  Min words/doc: {non_zero_counts.min()}")
print(f"  Max words/doc: {non_zero_counts.max()}")
print(f"  Median words/doc: {pd.Series(non_zero_counts).median():.2f}")

print("\n" + "="*80)
print("BAG OF WORDS VECTORIZATION")
print("="*80)
count_vectorizer = CountVectorizer(
    stop_words='english',
    min_df=5,
    max_df=0.9
)

bow_matrix = count_vectorizer.fit_transform(corpus)
bow_feature_names = count_vectorizer.get_feature_names_out()

print(f"\nMatrix shape: {bow_matrix.shape}")
print(f"Vocabulary sample: {list(bow_feature_names[:10])}")

word_frequencies = bow_matrix.sum(axis=0).A1
top_indices = word_frequencies.argsort()[-20:][::-1]

print(f"\nTop 20 most frequent words:")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. {bow_feature_names[idx]:20s} - {int(word_frequencies[idx]):6d} occurrences")

bow_non_zero = (bow_matrix != 0).sum(axis=1).A1
total_word_counts = bow_matrix.sum(axis=1).A1

print(f"\nSummary:")
print(f"  Vocabulary size: {len(bow_feature_names)}")
print(f"  Avg unique words/doc: {bow_non_zero.mean():.2f}")
print(f"  Median unique words/doc: {pd.Series(bow_non_zero).median():.2f}")
print(f"  Avg total words/doc: {total_word_counts.mean():.2f}")
print(f"  Total words (all docs): {int(total_word_counts.sum())}")
