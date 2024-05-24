#==============================================================================================================================================
#Loading the core packages
#==============================================================================================================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import pickle



#==============================================================================================================================================
#Loading the data
#==============================================================================================================================================

def load_data(data):
    df = pd.read_csv(data)
    return df

candidate_df = load_data("data/candidate2.csv")
master_program_df = load_data("data/Master_Program.csv")


#==============================================================================================================================================
#Preprocessing for individual recommendation
#==============================================================================================================================================

# Handle missing values by filling them with empty strings
candidate_df['Combined_Info'] = candidate_df['Current_Job'].fillna('') + ' ' + candidate_df['Career_Interest'].fillna('') + ' ' + candidate_df['Qualification'].fillna('')
master_program_df['Combined_Info'] = master_program_df['Program_Name'].fillna('') + ' ' + master_program_df['Program_Description'].fillna('') + ' ' + master_program_df['Program_Requirements'].fillna('')

# Combine text data from both datasets for fitting the TF-IDF vectorizer
combined_info = pd.concat([candidate_df['Combined_Info'], master_program_df['Combined_Info']])

# Fit the TF-IDF vectorizer on the combined text data
vectorizer = TfidfVectorizer()
vectorizer.fit(combined_info)

# Transform the individual datasets
candidate_tfidf = vectorizer.transform(candidate_df['Combined_Info'])
program_tfidf = vectorizer.transform(master_program_df['Combined_Info'])

# Save the model
with open('recommender_model.pkl', 'wb') as file:
    pickle.dump((vectorizer, program_tfidf, master_program_df), file)