# 311 Service Requests - Advanced Topic Analysis

## Overview
This project analyzes New York City 311 service request data to identify common complaint themes using Natural Language Processing (NLP) and unsupervised machine learning. It processes raw complaint descriptions, converts them into numerical representations, and applies topic modeling and clustering techniques to uncover patterns and trends in municipal service issues.

## Approach
Dual-method NLP pipeline comparing LDA topic modelling and K-Means clustering on the same dataset to validate pattern consistency:
- Text preprocessing → Bag of Words (LDA) and TF-IDF + TruncatedSVD (K-Means)
- TruncatedSVD reduced TF-IDF matrix to 50 components, retaining 83.08% of variance
- Both methods tuned to 10 topics/clusters

## Features
- Automated text preprocessing (cleaning, stopword removal, lemmatization)
- Text vectorization using Bag-of-Words and TF-IDF with n-grams
- Topic modeling with Latent Dirichlet Allocation (LDA)
- Clustering with K-Means (with dimensionality reduction via TruncatedSVD)
- Model evaluation using perplexity, silhouette score, and Davies-Bouldin index
- Visualizations including word clouds, t-SNE plots, and distribution charts
- Cross-analysis comparing LDA topics and K-Means clusters

## Results

**LDA Topic Modelling**
- Perplexity: 62.38 (good probabilistic fit)
- UMass Coherence: -3.9881 (meaningful, interpretable topics)
- 10 coherent topics extracted including: loud music/parties, 
  blocked hydrants, building-wide complaints, parking violations, 
  noise disturbances

**K-Means Clustering**
- Silhouette Score: 0.6337
- Davies-Bouldin Index: 0.6847 (strong, well-separated clusters)
- 6 semantically labelled clusters confirmed including: pest/trash/mould, 
  parking violations, blocked hydrants, loud music

**Key finding:** Both methods independently identified the same 
structural patterns confirming genuine semantic consistency 
in the dataset rather than artefacts of either method.

## Methodological Note
LDA is preferable for understanding mixed themes within a document. 
K-Means suits segmentation and complaint routing workflows requiring 
single-label assignment per record. Using both strengthens confidence 
in the discovered patterns.

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
