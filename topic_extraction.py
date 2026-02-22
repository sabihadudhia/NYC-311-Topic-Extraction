import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("="*80)
print("TOPIC EXTRACTION & CLUSTERING")
print("="*80)

def parse_topic_range():
    parser = argparse.ArgumentParser(description="Topic extraction and clustering")
    parser.add_argument(
        "--topic-range",
        type=str,
        default="5,7,10",
        help="Comma-separated list of topic counts to evaluate for LDA (e.g., 5,8,12,15)"
    )
    parser.add_argument(
        "--cluster-range",
        type=str,
        default="5,7,10",
        help="Comma-separated list of cluster counts to evaluate for K-Means (e.g., 4,6,8,10)"
    )
    args = parser.parse_args()

    try:
        topic_range = [int(value.strip()) for value in args.topic_range.split(',') if value.strip()]
    except ValueError:
        raise ValueError("Invalid --topic-range. Use comma-separated integers, e.g., 5,7,10")

    try:
        cluster_range = [int(value.strip()) for value in args.cluster_range.split(',') if value.strip()]
    except ValueError:
        raise ValueError("Invalid --cluster-range. Use comma-separated integers, e.g., 5,7,10")

    if not topic_range:
        raise ValueError("--topic-range must contain at least one integer")
    if any(topic <= 1 for topic in topic_range):
        raise ValueError("All topic counts in --topic-range must be greater than 1")
    if not cluster_range:
        raise ValueError("--cluster-range must contain at least one integer")
    if any(cluster <= 1 for cluster in cluster_range):
        raise ValueError("All cluster counts in --cluster-range must be greater than 1")

    return sorted(set(topic_range)), sorted(set(cluster_range))

topic_range, cluster_range = parse_topic_range()
print(f"Using topic range: {topic_range}")
print(f"Using cluster range: {cluster_range}")

df = pd.read_csv(r"C:\Users\sabih\OneDrive\Desktop\Project - Data Analysis\preprocessed_data.csv")
corpus = df['Problem Detail (formerly Descriptor)'].fillna('').astype(str).values

print(f"\nLoaded {len(corpus)} documents")

print("\n" + "="*80)
print("VECTORIZATION")
print("="*80)
count_vectorizer = CountVectorizer(
    stop_words='english',
    min_df=5,
    max_df=0.9
)
bow_matrix = count_vectorizer.fit_transform(corpus)
bow_feature_names = count_vectorizer.get_feature_names_out()

# For K-Means: Use TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"Bag of Words matrix shape: {bow_matrix.shape}")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# ============================================================================
# --- 2. LATENT DIRICHLET ALLOCATION (LDA) TOPIC MODELING ---
# ============================================================================

print("\n" + "="*80)
print("LATENT DIRICHLET ALLOCATION (LDA) - TOPIC MODELING")
print("="*80)

def compute_umass_coherence(model, doc_term_matrix, n_top_words=10):
    binary_matrix = (doc_term_matrix > 0).astype(np.int64)
    doc_frequencies = np.asarray(binary_matrix.sum(axis=0)).ravel()
    co_occurrence = (binary_matrix.T @ binary_matrix).toarray()

    topic_scores = []
    for topic in model.components_:
        top_indices = topic.argsort()[-n_top_words:][::-1]
        pair_scores = []

        for m in range(1, len(top_indices)):
            wi = top_indices[m]
            for l in range(m):
                wj = top_indices[l]
                co_count = co_occurrence[wi, wj]
                wj_count = doc_frequencies[wj]
                if wj_count > 0:
                    pair_scores.append(np.log((co_count + 1) / wj_count))

        if pair_scores:
            topic_scores.append(float(np.mean(pair_scores)))

    if not topic_scores:
        return float('-inf')

    coherence = float(np.mean(topic_scores))
    return coherence if np.isfinite(coherence) else float('-inf')

best_coherence = float('-inf')
best_perplexity = float('inf')
best_lda = None
best_n_topics = 0

print("\nTuning topic count using coherence...")
for n_topics in topic_range:
    print(f"\nTrying {n_topics} topics...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='batch',
        n_jobs=-1
    )
    lda_model.fit(bow_matrix)

    perplexity = lda_model.perplexity(bow_matrix)
    coherence = compute_umass_coherence(lda_model, bow_matrix, n_top_words=10)
    if not np.isfinite(coherence):
        coherence = float('-inf')
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  UMass coherence: {coherence:.4f}")

    if (coherence > best_coherence) or (np.isclose(coherence, best_coherence) and perplexity < best_perplexity):
        best_coherence = coherence
        best_perplexity = perplexity
        best_lda = lda_model
        best_n_topics = n_topics

print(f"\nSelected LDA model: {best_n_topics} topics (UMass coherence: {best_coherence:.4f}, perplexity: {best_perplexity:.2f})")

if best_lda is None:
    raise RuntimeError("LDA model selection failed. Check topic range and input data.")

lda_model = best_lda
n_topics = best_n_topics
lda_topics = lda_model.transform(bow_matrix)

print(f"\nCompleted")
print(f"  Log likelihood: {lda_model.score(bow_matrix):.2f}")
print(f"  Perplexity: {lda_model.perplexity(bow_matrix):.2f}")
print(f"  UMass coherence: {best_coherence:.4f}")

print("\n" + "-"*80)
print("TOP 10 WORDS PER TOPIC")
print("-"*80)

def display_topics(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        for word, weight in zip(top_words, top_weights):
            print(f"  {word:20s} - {weight:.4f}")

display_topics(lda_model, bow_feature_names, n_top_words=10)

print("\n" + "-"*80)
print("TOPIC DISTRIBUTION")
print("-"*80)
dominant_topics = lda_topics.argmax(axis=1)
df['LDA_Topic'] = dominant_topics + 1  # Add 1 to make topics 1-indexed

# Display topic distribution
topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
print("\nDocument count per topic:")
for topic_idx, count in topic_counts.items():
    print(f"  Topic {topic_idx + 1}: {count} documents ({count/len(corpus)*100:.2f}%)")

# Show examples from each topic
print("\n" + "-"*80)
print("SAMPLE DOCUMENTS FROM EACH TOPIC")
print("-"*80)

for topic_num in range(n_topics):
    print(f"\nTopic {topic_num + 1} - Sample documents:")
    topic_docs = df[df['LDA_Topic'] == topic_num + 1]['Problem Detail (formerly Descriptor)'].head(3)
    for idx, doc in enumerate(topic_docs, 1):
        print(f"  {idx}. {doc[:80]}...")

# ============================================================================
# --- 3. K-MEANS CLUSTERING ---
# ============================================================================

print("\n" + "="*80)
print("K-MEANS CLUSTERING")
print("="*80)

valid_cluster_range = [cluster for cluster in cluster_range if cluster < len(corpus)]
if not valid_cluster_range:
    raise ValueError("No valid cluster values in --cluster-range. Each value must be less than number of documents.")

best_silhouette = -1
best_kmeans = None
best_n_clusters = 0
best_labels = None

print("\nTuning cluster count using silhouette...")
for n_clusters in valid_cluster_range:
    print(f"\nTrying {n_clusters} clusters...")
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        max_iter=300,
        n_init=10
    )

    kmeans_labels = kmeans_model.fit_predict(tfidf_matrix)
    silhouette_avg = silhouette_score(tfidf_matrix, kmeans_labels)
    print(f"  Silhouette Score: {silhouette_avg:.4f}")

    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_kmeans = kmeans_model
        best_n_clusters = n_clusters
        best_labels = kmeans_labels

print(f"\nCompleted")
print(f"Selected K-Means model: {best_n_clusters} clusters (silhouette: {best_silhouette:.4f})")

kmeans_model = best_kmeans
n_clusters = best_n_clusters
kmeans_labels = best_labels

print("\n" + "-"*80)
print("EVALUATION METRICS")
print("-"*80)

if n_clusters > 1 and n_clusters < len(corpus):
    silhouette_avg = best_silhouette
    davies_bouldin = davies_bouldin_score(tfidf_matrix.toarray(), kmeans_labels)
    
    print(f"\nSilhouette Score: {silhouette_avg:.4f} (higher is better)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    print(f"Inertia: {kmeans_model.inertia_:.2f}")
df['KMeans_Cluster'] = kmeans_labels + 1  # Add 1 to make clusters 1-indexed

# Display cluster distribution
cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
print("\n" + "-"*80)
print("CLUSTER DISTRIBUTION")
print("-"*80)
print("\nDocument count per cluster:")
for cluster_idx, count in cluster_counts.items():
    print(f"  Cluster {cluster_idx + 1}: {count} documents ({count/len(corpus)*100:.2f}%)")

# --- Display Top Terms for Each Cluster
print("\n" + "-"*80)
print("TOP 10 TERMS FOR EACH CLUSTER")
print("-"*80)

# Get cluster centers (sorted by feature importance)
cluster_centers = kmeans_model.cluster_centers_

for cluster_idx in range(n_clusters):
    print(f"\nCluster {cluster_idx + 1}:")
    center = cluster_centers[cluster_idx]
    top_indices = center.argsort()[-10:][::-1]
    top_terms = [tfidf_feature_names[i] for i in top_indices]
    top_scores = [center[i] for i in top_indices]
    for term, score in zip(top_terms, top_scores):
        print(f"  {term:20s} - {score:.4f}")

# --- Show Sample Documents from Each Cluster
print("\n" + "-"*80)
print("SAMPLE DOCUMENTS FROM EACH CLUSTER")
print("-"*80)

for cluster_num in range(1, n_clusters + 1):
    print(f"\nCluster {cluster_num} - Sample documents:")
    cluster_docs = df[df['KMeans_Cluster'] == cluster_num]['Problem Detail (formerly Descriptor)'].head(3)
    for idx, doc in enumerate(cluster_docs, 1):
        print(f"  {idx}. {doc[:80]}...")

# ============================================================================
# --- 4. SAVE RESULTS ---
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save the dataframe with topic and cluster assignments
output_path = r"C:\Users\sabih\OneDrive\Desktop\Project - Data Analysis\topic_extraction_results.csv"
df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Added columns: 'LDA_Topic' and 'KMeans_Cluster'")

print("\n" + "="*80)
print("TOPIC EXTRACTION AND CLUSTERING COMPLETED SUCCESSFULLY!")
print("="*80)
