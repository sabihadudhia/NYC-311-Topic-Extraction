# 311 Service Requests - Advanced Topic Analysis

## Overview
This project analyzes New York City 311 service request data to identify common complaint themes using Natural Language Processing (NLP) and unsupervised machine learning. It processes raw complaint descriptions, converts them into numerical representations, and applies topic modeling and clustering techniques to uncover patterns and trends in municipal service issues.

## Features
- Automated text preprocessing (cleaning, stopword removal, lemmatization)
- Text vectorization using Bag-of-Words and TF-IDF with n-grams
- Topic modeling with Latent Dirichlet Allocation (LDA)
- Clustering with K-Means (with dimensionality reduction via TruncatedSVD)
- Model evaluation using perplexity, silhouette score, and Davies-Bouldin index
- Visualizations including word clouds, t-SNE plots, and distribution charts
- Cross-analysis comparing LDA topics and K-Means clusters

## Development Approach
Two-stage pipeline development: basic vectorisation 
(topic_extraction.py) followed by an advanced pipeline 
(advanced_topic_analysis.py) incorporating domain-specific 
stopwords, trigrams, and TruncatedSVD improving K-Means 
Silhouette Score from 0.5627 to 0.6337 and LDA coherence 
from -4.7819 to -3.9881.

## Results

**Script Comparison: Basic vs Advanced Pipeline**

| Metric | Basic Pipeline | Advanced Pipeline |
|---|---|---|
| LDA Coherence (UMass) | -4.7819 | -3.9881 |
| LDA Perplexity | 47.24 | 62.76 |
| K-Means Silhouette | 0.5627 | 0.6337 |
| K-Means Davies-Bouldin | 0.9252 | 0.6847 |

The advanced pipeline (domain-specific stopwords + trigrams + 
TruncatedSVD) outperformed the basic version on every meaningful 
metric. All results below are from the advanced pipeline.

---

**Dimensionality Reduction**
- TruncatedSVD reduced TF-IDF matrix to 50 components
- Retained 83.08% of explained variance

---
Distribution Plots: K-Means Clustering and LDA Topic Modeling

<img width="4170" height="1466" alt="distribution_plots" src="https://github.com/user-attachments/assets/86e40d96-61e4-4c03-9519-bd1e3e8468f9" />

---

**LDA Topic Modelling — 10 topics**
- Perplexity: 62.76
- UMass Coherence: -3.9881 (best across 5, 7, and 10 topic configurations)
- Dominant topic: Building-wide complaints (28.53%, 5,245 documents)

10 topics extracted:
1. Commercial vehicle & overnight parking
2. Double-parked & blocking traffic
3. Entire building complaints
4. Apartment complaints, banging/pounding, pest & mould
5. Loud music & parties, license plate violations
6. Water leaks & slow flow
7. Parking sign violations & illegal parking
8. Residential door, floor & gas complaints
9. Partial access issues & truck route violations
10. Blocked hydrants & sidewalks

---

**K-Means Clustering — 10 clusters**
- Silhouette Score: 0.6337 (best across 5, 7, and 10 cluster configurations)
- Davies-Bouldin Index: 0.6847 (lower is better — indicates well-separated clusters)
- Inertia: 7,427.16

10 clusters identified:
1. Pest, trash & mould (1,736 documents, 9.44%)
2. Entire building complaints (3,123 documents, 16.98%)
3. Apartment complaints (1,677 documents, 9.12%)
4. License plate violations (554 documents, 3.01%)
5. Loud music & parties (820 documents, 4.46%)
6. Banging & pounding noise (186 documents, 1.01%)
7. Blocked hydrants (227 documents, 1.23%)
8. Mixed complaints — sidewalk, license plate, noise (8,153 documents, 44.34%)
9. Parking sign violations (958 documents, 5.21%)
10. Blocked sidewalks (953 documents, 5.18%)

---
t-SNE visualization of 10 LDA topics (left) and 10 K-Means clusters (right) after TF-IDF vectorization and dimensionality reduction using Truncated SVD (50 components, 82.88% variance retained). The K-Means clusters show clearer spatial separation, supporting the silhouette score of 0.6337, while LDA topics exhibit more overlap due to their probabilistic nature.

<img width="4663" height="1769" alt="tsne_visualization" src="https://github.com/user-attachments/assets/e8e79dfd-b863-4904-8618-82d361c27af3" />

---
**Key Finding**
Strong cross-method consistency on dominant themes: building 
complaints, blocked hydrants, noise, and parking violations were 
independently identified by both LDA and K-Means, confirming 
genuine semantic structure in the data rather than artefacts of 
either algorithm.

Honest limitation: LDA Topics 1, 2, 6, and 8 collapsed into 
K-Means Cluster 8 (mixed complaints), revealing vocabulary overlap 
between minor complaint categories. This reflects a known trade-off: 
LDA captures mixed themes within documents while K-Means assigns 
a single label per record making K-Means less suited to 
complaint types with overlapping terminology.

---

## Methodological Note
LDA is preferable for understanding mixed themes within a document, 
useful for exploratory analysis and policy insight. K-Means suits 
complaint routing and triage workflows where each record requires a 
single category label. The convergence of both methods on major 
themes strengthens confidence in the discovered patterns. The 
perplexity increase from basic (47.24) to advanced (62.76) pipeline 
reflects a deliberate trade-off, sacrificing some predictive fit 
for significantly more coherent, human-interpretable topics.

## Technologies
- Python
- pandas
- scikit-learn
- NLTK
- matplotlib & seaborn
- WordCloud
- NumPy

## Setup / Installation
1. Clone or download the project:
```bash
cd "C:\Users\sabih\OneDrive\Desktop\Project - Data Analysis"
```
2. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
3. Install dependencies:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
```
4. Ensure your input data file is in place:
   - Input: `311_Service_Requests_from_2020_to_Present_20260215 (2).csv`

## Usage
Run the scripts in the following order:

### Step 1: Data Preprocessing
```bash
python data_preprocessing.py
```
**Output**: `preprocessed_data.csv` (cleaned dataset with removed duplicates and processed text)

### Step 2: Text Vectorization
```bash
python text_vectorization.py
```
**Output**: Console statistics on vocabulary size, word frequencies, and document characteristics

### Step 3: Topic Extraction & Clustering
```bash
python topic_extraction.py
```
**Output**: `topic_extraction_results.csv` (with LDA_Topic and KMeans_Cluster columns)

### Step 4: Advanced Analysis
```bash
python advanced_topic_analysis.py
```

## Project Structure
```
Project - Data Analysis/
├── data_preprocessing.py                    
├── text_vectorization.py                    
├── topic_extraction.py                      
├── advanced_topic_analysis.py               
├── README.md                                
├── 311_Service_Requests_from_2020_to_...csv 
├── preprocessed_data.csv                    
├── topic_extraction_results.csv             
├── advanced_topic_analysis_results.csv     
├── analysis_summary.csv
├── run_pipeline_and_save_output.py             
├── lda_wordclouds.png                       
├── tsne_visualization.png             
└── distribution_plots.png                  
```

## Outputs 
1. CSV Files with Assignments
2. Analysis Summary
3. Visualizations

## Notes / Additional Info
- The input dataset must contain a complaint description column (e.g., “Problem Detail (formerly Descriptor)”).
- Domain-specific stopwords are removed to improve topic quality.
- The model tests multiple topic/cluster numbers and selects the best configuration based on evaluation metrics.
- If NLTK resources are missing, they will be downloaded automatically.
- For large datasets, reduce max_features in vectorizers to improve performance. Consider domain-specific stopwords for your complaint category
