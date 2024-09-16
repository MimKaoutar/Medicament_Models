import pandas as pd
import os 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--test_ratio", type=int, default=0.2)
parser.add_argument("--val_ratio", type=int, default=0.01)
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
print("number of medicaments :", len(filtered_df.medicament.values))
healthy_df = sparse_df[sparse_df['diagnosis']==0]
defective_df = sparse_df[sparse_df['diagnosis']==1]
test_defective, train_defective = select(defective_df, filtered_df,args.test_ratio,healthy=False) 
val_defective, train_defective = select(train_defective, filtered_df,args.val_ratio,healthy=False) 
test_healthy , train_healthy = select(healthy_df, filtered_df, args.test_ratio) 
val_healthy , train_healthy = select(train_healthy, filtered_df, args.val_ratio) 

train = pd.concat([train_healthy,train_defective], ignore_index=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
val = pd.concat([val_healthy,val_defective], ignore_index=True)
val = val.sample(frac=1, random_state=42).reset_index(drop=True)
test = pd.concat([test_healthy,test_defective], ignore_index=True)
test = test.sample(frac=1, random_state=42).reset_index(drop=True)
train.drop(columns=['ID','fin_grossesse'], inplace=True)
val.drop(columns=['ID','fin_grossesse'], inplace=True)
test.drop(columns=['ID','fin_grossesse'], inplace=True)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index= False)
test.to_csv("test.csv", index=False)
print("number of train records : ", len(train))
print("Train count classes:", train.diagnosis.value_counts())
print("All meds exist", (train.sum() > 0).all())
print("number of  val records : ", len(val))
print("Validation count classes:", val.diagnosis.value_counts())
print("All meds exist", (val.sum() > 0).all())
print("number of test records :", len(test))
print("Test count classes:", test.diagnosis.value_counts())
print("All meds exist", (test.sum() > 0).all())