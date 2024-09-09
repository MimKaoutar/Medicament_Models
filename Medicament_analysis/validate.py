import pandas as pd

def validate(filtered_df, train, test, ratio):
    list_meds = filtered_df['medicament'].values

    # Separate the data by diagnosis
    train_healthy = train[train['diagnosis'] == 0]
    test_healthy = test[test['diagnosis'] == 0]
    train_defective = train[train['diagnosis'] == 1]
    test_defective = test[test['diagnosis'] == 1]

    # Define a helper function to check the condition
    def check_condition(df, ratio, column_name):
        for value in list_meds:
            medicament_sum = df["_"+value].sum()
            threshold = ratio * filtered_df.loc[filtered_df['medicament'] == value, column_name].values[0]
            if medicament_sum < threshold:
                print(medicament_sum)
                print(value)
                print(threshold)
                return False
        return True

    # Check conditions for all combinations
    if not (check_condition(test_healthy, ratio, "count_diagnosis_0") and
            check_condition(test_defective, ratio, "count_diagnosis_1")):
        return False

    return True

filtered_df= pd.read_csv("medicament_analysis.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(validate(filtered_df, train, test, 0.2))