import pandas as pd
import os 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--test_ratio", type=int, default=0.2)
args = parser.parse_args()
 
def select(sparse_df, filtered_df, ratio, healthy=True) -> pd.DataFrame:
    target = 0 if healthy else 1 
    filtered_df = filtered_df.sort_values(by=['count_diagnosis_'+str(target)], ascending=True)
    list_meds = list(filtered_df.medicament.values)
    sum_records= {value : 0 for value in list_meds}
    df = pd.DataFrame()
    
    for value in list_meds:
        limit = int(ratio * filtered_df.loc[filtered_df['medicament'] == value, "count_diagnosis_"+str(target)].values[0])
        if sum_records[value] < limit: 
            to_add = sum_records[value] - limit
            rows_to_add = sparse_df.loc[sparse_df["_"+value] == 1][:to_add]
            df = pd.concat([df , rows_to_add], ignore_index=True)
            for value in list_meds:
                sum_records[value] = df["_"+value].sum()
            index_to_remove = rows_to_add.index
            sparse_df = sparse_df.drop(index_to_remove)
    return df, sparse_df

filtered_df= pd.read_csv("medicament_analysis.csv")
sparse_df = pd.read_csv("sparse_med.csv")  
healthy_df = sparse_df[sparse_df['diagnosis']==0]
defective_df = sparse_df[sparse_df['diagnosis']==1]
test_defective, train_defective = select(defective_df, filtered_df,args.test_ratio,healthy=False) 
test_healthy , train_healthy = select(healthy_df, filtered_df, args.test_ratio) 

train = pd.concat([train_healthy,train_defective], ignore_index=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
test = pd.concat([test_healthy,test_defective], ignore_index=True)
test = test.sample(frac=1, random_state=42).reset_index(drop=True)
train.drop(columns=['ID','fin_grossesse'], inplace=True)
test.drop(columns=['ID','fin_grossesse'], inplace=True)
train.to_csv("train.csv")
test.to_csv("test.csv")
print("number of train records : ", len(train))
print("number of test records :", len(test))
