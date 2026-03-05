import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)

print("="*80)
print("DATA PREPROCESSING PIPELINE")
print("="*80)

file_path = r"C:\Users\sabih\OneDrive\Desktop\Project - Data Analysis\311_Service_Requests_from_2020_to_Present_20260215 (2).csv"
df = pd.read_csv(file_path)

print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn info:")
print(df.info())

print("\n" + "="*80)
print("STEP 1: COLUMN SELECTION")
print("="*80)
columns_to_remove = ['Unique Key', 'Closed Date', 'Agency', 'Agency Name', 'Additional Details', 'Location Type', 
                     'Incident Zip', 'Incident Address', 'Street Name', 'Cross Street 1', 'Cross Street 2', 'Intersection Street 1', 
                     'Intersection Street 2', 'Address Type', 'City', 'Landmark', 'Facility Type', 'Due Date', 
                     'Resolution Action Updated Date', 'Community Board', 'Council District', 'Police Precinct', 'BBL', 
                     'X Coordinate (State Plane)', 'Y Coordinate (State Plane)', 'Open Data Channel Type', 'Park Facility Name', 
                     'Park Borough', 'Vehicle Type', 'Taxi Company Borough', 'Taxi Pick Up Location', 'Bridge Highway Name', 
                     'Bridge Highway Direction', 'Road Ramp', 'Bridge Highway Segment', 'Latitude', 'Longitude', 'Location']
df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

print(f"Retained {len(df.columns)} columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

print("\n" + "="*80)
print("STEP 2: LOWERCASE CONVERSION")
print("="*80)
created_date = df['Created Date'].copy()

# Apply the lowercase operation to all columns except Created Date
for col in df.columns:
    if col != 'Created Date':
        df[col] = df[col].astype(str).str.lower()

# Restore the original Created Date column
df['Created Date'] = created_date

print("Data has been converted to lowercase")

# --- 3. Removal of punctuation and numbers 
print("Removing punctuation and numbers from data...")

for col in df.columns:
    if col != 'Created Date':
        # Remove all punctuation and numbers **except slash**
        df[col] = df[col].replace(r'[^a-zA-Z\s/]+', '', regex=True)


print("Punctuation and numbers have been removed from the data (except slashes)")

# --- 4. Removal of Stop Words 
print("Removing stop words...")

# Define the set of English stopwords for faster lookups
stop_words = set(stopwords.words('english'))

# Define a function to remove stopwords from a single string
def remove_stopwords(text):
    # Convert to lowercase and split the text into words (tokenization)
    words = text.lower().split()
    # Filter out words that are in the stop_words set using a list comprehension
    filtered_words = [word for word in words if word not in stop_words]
    # Join the remaining words back into a single string
    return ' '.join(filtered_words)

for col in df.columns:
    if col != 'Created Date':
        df[col] = df[col].apply(remove_stopwords)

print("Stop words have been removed")

# --- 5. Tokenization
print("Performing tokenization (splitting text into words)...")

def tokenize_text(text):
    # Split text into tokens (words)
    tokens = text.split()
    return tokens

# Apply tokenization and rejoin for storage (to keep in tabular format)
for col in df.columns:
    if col != 'Created Date':
        df[col] = df[col].apply(lambda x: ' '.join(tokenize_text(x)))

print("Tokenization completed")

# --- 6. Lemmatization
print("Performing lemmatization...")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to lemmatize text
def lemmatize_text(text):
    # Split text into tokens
    tokens = text.split()
    # Lemmatize each token and join back
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Apply lemmatization to all columns except Created Date
for col in df.columns:
    if col != 'Created Date':
        df[col] = df[col].apply(lemmatize_text)

print("Lemmatization completed")

# --- 7. Duplicate Removal
print("Removing duplicate rows...")
initial_rows = len(df)
df = df.drop_duplicates()
removed_duplicates = initial_rows - len(df)
print(f"Removed {removed_duplicates} duplicate rows")
print(f"Remaining rows: {len(df)}")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nFirst 5 rows:")
print(df.head())

print("\n" + "="*80)
print("SAVING OUTPUT")
print("="*80)

output_file_path = r"C:\Users\sabih\OneDrive\Desktop\Project - Data Analysis\preprocessed_data.csv"
df.to_csv(output_file_path, index=False)
print(f"\nSaved to: {output_file_path}")
print("\nPreprocessing completed successfully!")