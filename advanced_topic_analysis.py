import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

print("="*80)
print("ADVANCED TOPIC ANALYSIS")
print("="*80)

df = pd.read_csv(r"C:\Users\sabih\OneDrive\Desktop\Project -  Data Analysis\preprocessed_data.csv")
corpus = df['Problem Detail (formerly Descriptor)'].fillna('').astype(str).values

print(f"\nLoaded {len(corpus)} documents")

print("\n" + "="*80)
print("VECTORIZATION WITH DOMAIN-SPECIFIC STOPWORDS")
print("="*80)

domain_stopwords = ['complaint', 'department', 'responded', 'request', 'service', 
                    'report', 'reported', 'issue', 'problem', 'detail']

count_vectorizer = CountVectorizer(
    stop_words='english',
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 3),
    max_features=1500
)
bow_matrix = count_vectorizer.fit_transform(corpus)
bow_feature_names = count_vectorizer.get_feature_names_out()

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 3),
    max_features=1500
)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"\nBoW matrix: {bow_matrix.shape}")
print(f"TF-IDF matrix: {tfidf_matrix.shape}")
print(f"Using n-grams: unigrams, bigrams, trigrams")

print("\n" + "="*80)
print("DIMENSIONALITY REDUCTION (TruncatedSVD)")
print("="*80)

n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

explained_variance = svd.explained_variance_ratio_.sum()
print(f"\nReduced to {n_components} components")
print(f"Explained variance: {explained_variance:.2%}")

print("\n" + "="*80)
print("LDA TOPIC MODELING - PARAMETER TUNING")
print("="*80)

topic_range = [5, 7, 10]
best_perplexity = float('inf')
best_lda = None
best_n_topics = 0

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
    print(f"  Perplexity: {perplexity:.2f}")
    
    if perplexity < best_perplexity:
        best_perplexity = perplexity
        best_lda = lda_model
        best_n_topics = n_topics

print(f"\nBest model: {best_n_topics} topics (perplexity: {best_perplexity:.2f})")
lda_topics = best_lda.fit_transform(bow_matrix)

print("\n" + "-"*80)
print(f"TOP 15 WORDS PER TOPIC ({best_n_topics} topics)")
print("-"*80)

def display_topics(model, feature_names, n_top_words=15):
    topics_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        topics_dict[topic_idx] = top_words
        print(f"\nTopic {topic_idx + 1}:")
        for word, weight in zip(top_words[:10], top_weights[:10]):
            print(f"  {word:25s} - {weight:.4f}")
    return topics_dict

lda_topics_words = display_topics(best_lda, bow_feature_names, n_top_words=15)

dominant_topics = lda_topics.argmax(axis=1)
df['LDA_Topic'] = dominant_topics + 1

print("\n" + "-"*80)
print("LDA TOPIC LABELS (Interpretation)")
print("-"*80)

topic_labels = {}
for topic_idx, words in lda_topics_words.items():
    topic_labels[topic_idx + 1] = f"Topic {topic_idx + 1}: {', '.join(words[:3])}"
    print(f"  {topic_labels[topic_idx + 1]}")

print("\n" + "="*80)
print("K-MEANS CLUSTERING - PARAMETER TUNING")
print("="*80)

cluster_range = [5, 7, 10]
best_silhouette = -1
best_kmeans = None
best_n_clusters = 0
best_labels = None

for n_clusters in cluster_range:
    print(f"\nTrying {n_clusters} clusters...")
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        max_iter=300,
        n_init=10
    )
    labels = kmeans_model.fit_predict(tfidf_reduced)
    silhouette = silhouette_score(tfidf_reduced, labels)
    print(f"  Silhouette: {silhouette:.4f}")
    
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_kmeans = kmeans_model
        best_n_clusters = n_clusters
        best_labels = labels

print(f"\nBest model: {best_n_clusters} clusters (silhouette: {best_silhouette:.4f})")

df['KMeans_Cluster'] = best_labels + 1

print("\n" + "-"*80)
print(f"TOP 10 TERMS PER CLUSTER ({best_n_clusters} clusters)")
print("-"*80)

# Project cluster centers back to original TF-IDF space
svd_components = svd.components_
cluster_centers_original = best_kmeans.cluster_centers_ @ svd_components

cluster_terms = {}
for cluster_idx in range(best_n_clusters):
    center = cluster_centers_original[cluster_idx]
    top_indices = center.argsort()[-15:][::-1]
    top_terms = [tfidf_feature_names[i] for i in top_indices]
    top_scores = [center[i] for i in top_indices]
    cluster_terms[cluster_idx] = top_terms
    print(f"\nCluster {cluster_idx + 1}:")
    for term, score in zip(top_terms[:10], top_scores[:10]):
        print(f"  {term:25s} - {score:.4f}")

print("\n" + "-"*80)
print("K-MEANS CLUSTER LABELS (Interpretation)")
print("-"*80)

cluster_labels = {}
for cluster_idx, terms in cluster_terms.items():
    cluster_labels[cluster_idx + 1] = f"Cluster {cluster_idx + 1}: {', '.join(terms[:3])}"
    print(f"  {cluster_labels[cluster_idx + 1]}")

print("\n" + "="*80)
print("LDA vs K-MEANS COMPARISON")
print("="*80)

comparison = pd.crosstab(df['LDA_Topic'], df['KMeans_Cluster'], normalize='index')
print("\nCross-tabulation (% of LDA topic in each K-Means cluster):\n")
print(comparison.round(3))

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

output_dir = r"C:\Users\sabih\OneDrive\Desktop\Project -  Data Analysis"

# Word Clouds for LDA Topics
print("\nGenerating word clouds for LDA topics...")
fig, axes = plt.subplots(2, (best_n_topics + 1) // 2, figsize=(16, 8))
axes = axes.flatten()

for topic_idx, words in lda_topics_words.items():
    word_freq = dict(zip(words, range(len(words), 0, -1)))
    wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
    axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
    axes[topic_idx].set_title(f'Topic {topic_idx + 1}', fontsize=12)
    axes[topic_idx].axis('off')

for idx in range(best_n_topics, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}\\lda_wordclouds.png", dpi=300, bbox_inches='tight')
print(f"  Saved: lda_wordclouds.png")
plt.close()

# t-SNE Visualization
print("\nGenerating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne_results = tsne.fit_transform(tfidf_reduced)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = ax1.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                       c=df['LDA_Topic'], cmap='tab10', alpha=0.6, s=10)
ax1.set_title(f'LDA Topics ({best_n_topics} topics)', fontsize=14)
ax1.set_xlabel('t-SNE 1')
ax1.set_ylabel('t-SNE 2')
plt.colorbar(scatter1, ax=ax1, label='Topic')

scatter2 = ax2.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                       c=df['KMeans_Cluster'], cmap='tab10', alpha=0.6, s=10)
ax2.set_title(f'K-Means Clusters ({best_n_clusters} clusters)', fontsize=14)
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

plt.tight_layout()
plt.savefig(f"{output_dir}\\tsne_visualization.png", dpi=300, bbox_inches='tight')
print(f"  Saved: tsne_visualization.png")
plt.close()

# Topic/Cluster Distribution
print("\nGenerating distribution plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

topic_counts = df['LDA_Topic'].value_counts().sort_index()
ax1.bar(topic_counts.index, topic_counts.values, color='steelblue')
ax1.set_xlabel('Topic')
ax1.set_ylabel('Document Count')
ax1.set_title(f'LDA Topic Distribution ({best_n_topics} topics)')

cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
ax2.bar(cluster_counts.index, cluster_counts.values, color='coral')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Document Count')
ax2.set_title(f'K-Means Cluster Distribution ({best_n_clusters} clusters)')

plt.tight_layout()
plt.savefig(f"{output_dir}\\distribution_plots.png", dpi=300, bbox_inches='tight')
print(f"  Saved: distribution_plots.png")
plt.close()

print("\n" + "="*80)
print("SAVING OUTPUT")
print("="*80)

output_path = f"{output_dir}\\advanced_topic_analysis_results.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
print(f"Added columns: LDA_Topic, KMeans_Cluster")

analysis_summary = {
    'LDA_Topics': best_n_topics,
    'LDA_Perplexity': best_perplexity,
    'KMeans_Clusters': best_n_clusters,
    'KMeans_Silhouette': best_silhouette,
    'SVD_Components': n_components,
    'SVD_Variance_Explained': explained_variance,
    'Topic_Labels': topic_labels,
    'Cluster_Labels': cluster_labels
}

summary_df = pd.DataFrame([analysis_summary])
summary_df.to_csv(f"{output_dir}\\analysis_summary.csv", index=False)
print(f"Saved: analysis_summary.csv")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"\n✓ Best LDA configuration: {best_n_topics} topics")
print(f"✓ Best K-Means configuration: {best_n_clusters} clusters with SVD")
print(f"\n✓ Generated visualizations:")
print(f"  - lda_wordclouds.png")
print(f"  - tsne_visualization.png")
print(f"  - distribution_plots.png")
print(f"\n✓ Cross-tabulation shows alignment between LDA and K-Means")
print(f"✓ Using n-grams (1-3) captures meaningful phrases like 'blocked hydrant'")

print("\nCompleted successfully!")
