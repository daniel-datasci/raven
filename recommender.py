import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_programs(current_job, career_interest, qualification, vectorizer, program_tfidf, master_program_df):
    input_str = current_job + ' ' + career_interest + ' ' + qualification
    input_tfidf = vectorizer.transform([input_str])
    sim_scores = cosine_similarity(input_tfidf, program_tfidf).flatten()
    top_indices = sim_scores.argsort()[-3:][::-1]
    
    recommendations = master_program_df.iloc[top_indices][['Program_Name', 'Faculty', 'Job_Prospects', 'Program_Description', 'Program_Requirements', 'Tuition', 'Tuition_Coverage', 'Deadline_Applications_for_Autumn_2025', 'Deadline_Applications_for_Spring_2025']]
    return recommendations