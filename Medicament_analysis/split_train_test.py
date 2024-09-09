import pandas as pd
import os 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument("--test_ratio", type=int, default=0.8)
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
df_count.to_csv("medicament_analysis.csv", index =False)
print('Number of medicaments before filtering:', len(df_count))
    
#Applying threshold 
thres = args.threshold
filtered_df = df_count[df_count['count_diagnosis_1'] >= thres]
filtered_df['proba_H/M'] = filtered_df['count_diagnosis_0'] / (filtered_df['count_diagnosis_0'] + filtered_df['count_diagnosis_1'])
total_sum_diagnosis_0 = filtered_df['count_diagnosis_0'].sum()  # Total sum of 'count_diagnosis_0'
filtered_df['proba_H/NM'] = filtered_df['count_diagnosis_0'] / (total_sum_diagnosis_0 - filtered_df['count_diagnosis_0'])
print('Number of medicament after filtering:', len(filtered_df))
print('Minimum drug usage in healthy cases :', filtered_df.count_diagnosis_0.min())

#get combinaison matrix of filtered medicaments   
df = df[df['medicament'].isin(filtered_df.medicament)]
sparse_df = pd.get_dummies(df, columns=['medicament'], sparse=True, prefix='', dtype=int)
sparse_df = sparse_df.groupby(['ID','fin_grossesse','diagnosis']).sum().reset_index()
print('Number of records after applying threshold:', len(sparse_df))
sparse_df.to_csv("sparse_med.csv", index=False)   


def select(sparse_df, filtered_df, ratio, healthy=True) -> pd.DataFrame:
    target = 0 if healthy else 1 
    filtered_df = filtered_df.sort_values(by=['count_diagnosis'+target], ascending=False)
    list_meds = list(filtered_df.medicament.values)
    sum_records= {value : 0 for value in list_meds}
    df = pd.DataFrame()
    
    for value in list_meds:
        limit = ratio * filtered_df.loc[df['medicament'] == value, "count_diagnosis"+target]
        if sum_records[value] < limit: 
            to_add = sum_records[value] - limit
            rows_to_add = sparse_df.loc[sparse_df[value] == 1][:to_add]
            df = pd.concat([df , rows_to_add], ignore_index=True)
            for value in list_meds:
                sum_records[value] = df[value].sum()
            ids_to_remove = rows_to_add['ID']
            sparse_df = sparse_df[~sparse_df['ID'].isin(ids_to_remove)]
    return df, sparse_df


healthy_df = sparse_df[sparse_df['diagnosis']==0]
defective_df = sparse_df[sparse_df['diagnosis']==1]
test_defective, train_defective = select(sparse_df, filtered_df, args.test_ratio) 
test_healthy , train_healthy = select(sparse_df, filtered_df, args.test_ratio) 

train = pd.concat([train_healthy,train_healthy], ignore_index=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
test = pd.concat([test_healthy,test_healthy], ignore_index=True)
test = test.sample(frac=1, random_state=42).reset_index(drop=True)

print("number of train records : ", len(train))
print("number of test records :", len(test))


def validate(filtered_df, train, test, ratio):
    list_meds = filtered_df['medicament'].values

    # Separate the data by diagnosis
    train_healthy = train[train['diagnosis'] == 0]
    test_healthy = test[test['diagnosis'] == 0]
    train_defective = train[train['diagnosis'] == 1]
    test_defective = test[test['diagnosis'] == 1]

    # Define a helper function to check the condition
    def check_condition(df, column_name):
        for value in list_meds:
            medicament_sum = df[value].sum()
            threshold = ratio * filtered_df.loc[filtered_df['medicament'] == value, column_name]
            if medicament_sum <= threshold:
                return False
        return True

    # Check conditions for all combinations
    if not (check_condition(train_healthy, "count_diagnosis_0") and
            check_condition(test_healthy, "count_diagnosis_0") and
            check_condition(train_defective, "count_diagnosis_1") and
            check_condition(test_defective, "count_diagnosis_1")):
        return False

    return True
        





