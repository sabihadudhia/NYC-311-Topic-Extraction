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
