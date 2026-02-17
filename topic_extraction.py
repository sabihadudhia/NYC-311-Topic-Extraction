import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("="*80)
print("TOPIC EXTRACTION & CLUSTERING")
print("="*80)

df = pd.read_csv(r"C:\Users\sabih\OneDrive\Desktop\Project -  Data Analysis\preprocessed_data.csv")
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

n_topics = 10

print(f"\nFitting LDA model with {n_topics} topics...")
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='batch',
    n_jobs=-1
)

lda_topics = lda_model.fit_transform(bow_matrix)

print(f"\nCompleted")
print(f"  Log likelihood: {lda_model.score(bow_matrix):.2f}")
print(f"  Perplexity: {lda_model.perplexity(bow_matrix):.2f}")

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

n_clusters = 10

print(f"\nFitting K-Means model with {n_clusters} clusters...")
kmeans_model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    max_iter=300,
    n_init=10
)

kmeans_labels = kmeans_model.fit_predict(tfidf_matrix)

print(f"\nCompleted")

print("\n" + "-"*80)
print("EVALUATION METRICS")
print("-"*80)

if n_clusters > 1 and n_clusters < len(corpus):
    silhouette_avg = silhouette_score(tfidf_matrix, kmeans_labels)
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
output_path = r"C:\Users\sabih\OneDrive\Desktop\Project -  Data Analysis\topic_extraction_results.csv"
df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Added columns: 'LDA_Topic' and 'KMeans_Cluster'")

print("\n" + "="*80)
print("TOPIC EXTRACTION AND CLUSTERING COMPLETED SUCCESSFULLY!")
print("="*80)
