import pandas as pd
import os 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--threshold", type=int, default=30)
args = parser.parse_args()

df_count = pd.DataFrame()
#read and clean data
df = pd.read_csv("final_table.csv")
print("Number of records before cleaning: ", len(df))
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print("Number of records after cleaning:", len(df))

#Get usage stats
selected_columns = ['medicament','diagnosis']
df_count = df[selected_columns].groupby(['medicament'])['diagnosis'].value_counts().unstack(fill_value=0).reset_index()
df_count.columns = ['medicament', 'count_diagnosis_0', 'count_diagnosis_1']
df_count['total_records'] = df_count['count_diagnosis_0'] + df_count['count_diagnosis_1']
print('Number of medicaments before filtering:', len(df_count))
    
#Applying threshold 
thres = args.threshold
filtered_df = df_count[df_count['count_diagnosis_1'] >= thres]
filtered_df['proba_H/M'] = filtered_df['count_diagnosis_0'] / (filtered_df['count_diagnosis_0'] + filtered_df['count_diagnosis_1'])
total_sum_diagnosis_0 = filtered_df['count_diagnosis_0'].sum()  # Total sum of 'count_diagnosis_0'
filtered_df['proba_H/NM'] = filtered_df['count_diagnosis_0'] / (total_sum_diagnosis_0 - filtered_df['count_diagnosis_0'])
filtered_df.to_csv("medicament_analysis.csv", index =False)
print('Number of medicament after filtering:', len(filtered_df))
print('Minimum drug usage in healthy cases :', filtered_df.count_diagnosis_0.min())

#get combinaison matrix of filtered medicaments   
df = df[df['medicament'].isin(filtered_df.medicament)]
sparse_df = pd.get_dummies(df, columns=['medicament'], sparse=True, prefix='', dtype=int)
sparse_df = sparse_df.groupby(['ID','fin_grossesse','diagnosis']).sum().reset_index()
print('Number of records after applying threshold:', len(sparse_df))
sparse_df.to_csv("sparse_med.csv", index=False)   

        





