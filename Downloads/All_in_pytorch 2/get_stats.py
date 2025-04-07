import pandas as pd
import numpy as np
import json
import os 
from argparse import ArgumentParser
import matplotlib.cm as cm
import mpl_config as lmc
import matplotlib.pyplot as plt

#os.environ["PATH"] += os.pathsep + 'C:/Users/NOHGAB00/AppData/Local/Programs/MiKTeX/miktex/bin/x64'
lmc.initialize()

#initialize figures
fig = plt.figure(figsize=(10,6))
fig.set_size_inches((6.4,4))

#add threshold argument 
parser = ArgumentParser()
parser.add_argument("--thres", type=int, default=0) # 0 for now 
parser.add_argument("--save_dir", type=str, default='Results')
parser.add_argument("--save_data", type=str, default='Data')
parser.add_argument("--mode", type=str, default='thres')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.save_data, exist_ok=True)

#get probabilities (no thres)
def get_analysis(df):
    filename = "medicament_probabilities.csv"
    save_path = os.path.join(args.save_data, filename)

    D, J = df.shape[0] , df.shape[1]-1
    df_analysis = pd.DataFrame()
    list_drugs = df.columns[1:]
    df_analysis['medicament'] = list_drugs

    count_0 = ((df.iloc[:, 0] == 0).values[:, None] * (df.iloc[:, 1:J+1] == 1)).sum(axis=0)
    count_1 = ((df.iloc[:, 0] == 1).values[:, None] * (df.iloc[:, 1:J+1] == 1)).sum(axis=0)

    df_analysis["count_diagnosis_0"]= count_0.tolist()
    df_analysis["count_diagnosis_1"]= count_1.tolist()

    probabilities_columns =  [["proba_healthy_nodrug","proba_healthy_drug"],["proba_nothealthy_nodrug","proba_nothealthy_drug"]]
    for i in range(2):
        for j in range(2):
            numerator_test = ((df.iloc[:, 0] == i).values[:, None] * (df.iloc[:, 1:J+1] == j)).sum(axis=0)
            denominator_test = (df.iloc[:, 1:J+1] == j).sum(axis=0)
            result_test = numerator_test / denominator_test
            df_analysis[probabilities_columns[i][j]] = result_test.tolist()


    df_analysis.to_csv(save_path, index=False)
    return df_analysis

#filter drugs
def get_filtered_meds(df_analysis):
    filename1 = "filetered_probabilities.csv"
    filename2 = "removed_probabilities.csv"
    filename3 = "end_probabilities.csv"

    #get dangerous medicament
    df_results = pd.DataFrame()
    selected_columns = ['medicament','proba_nothealthy_nodrug','proba_nothealthy_drug','count_diagnosis_0','count_diagnosis_1']
    df_analysis = df_analysis.sort_values(by=['proba_nothealthy_drug'],ascending=False)
    df_analysis['total_diagnosis'] = df_analysis['count_diagnosis_0'] + df_analysis['count_diagnosis_1']

    #get safe medicament
    df_end = df_analysis[(df_analysis["proba_nothealthy_nodrug"]>df_analysis["proba_nothealthy_drug"])]
    #df_end = df_end[df_end['total_diagnosis']<= args.thres]

    #apply thereshold after probabilities
    df_results = df_analysis[(df_analysis["proba_nothealthy_nodrug"]<=df_analysis["proba_nothealthy_drug"])]
    df_kept = df_results[df_results['total_diagnosis']>= args.thres]
    df_removed = df_results[df_results['total_diagnosis']< args.thres]
    
    
    df_kept = df_kept[selected_columns].round(2)
    df_removed = df_removed[selected_columns].round(2)
    df_end = df_end[selected_columns].round(2)
    df_kept.reset_index(drop=True, inplace=True)
    df_removed.reset_index(drop=True, inplace=True)
    df_end.reset_index(drop=True, inplace=True)
    
    df_kept.to_csv(os.path.join(args.save_data, filename1), index=False)
    df_removed.to_csv(os.path.join(args.save_data, filename2), index=False)
    df_end.to_csv(os.path.join(args.save_data, filename3), index=False)

    return df_removed , df_kept, df_end

#Remove ID of removed drugs 
def remove_ID(drug_to_remove, df): 
    dict = {}
    df_copy = df
    for drug in drug_to_remove:
        df_cleaned = df[~(df[drug] == 1)]
        dict[drug.replace('_','')] = len(df_copy[(df_copy[drug] == 1)])
        df = df_cleaned.drop(columns=[drug])
    return dict, df



#get combinaison statistics
def get_comb_stats(df_results):
    filename = "Nbcombinaisons_" + args.mode +".pgf"
    save_path = os.path.join(args.save_dir, filename)

    df = pd.read_csv("None_None\pharma_table.csv")
    df = df[['ID', 'fin_grossesse','medicament','diagnosis']]
    df.drop_duplicates(inplace=True)
    df['medicament'] = "_"+df['medicament']
    df = df[df['medicament'].isin(df_results.medicament)]
    df1 = df[['ID', 'fin_grossesse','diagnosis']]
    df_comb = df1.groupby(['ID', 'fin_grossesse','diagnosis']).size().reset_index(name='counts')
    df_comb.sort_values(by=['counts'],inplace=True)

    class_0 = [x for x in df_comb[df_comb['diagnosis'] == 0]['counts']]
    class_1 = [x for x in df_comb[df_comb['diagnosis'] == 1]['counts']]
    
    _ , bins, _ = plt.hist([class_1,class_0], bins= len(df_comb['counts'].value_counts()),
                        histtype="bar", color=['lightseagreen','teal'], 
                        label=["grossesse avec malformation","grossesse sans malformation"])
    
    plt.xticks(((bins[:-1] + bins[1:])/2),np.arange(1,17))
    plt.xlabel("Nombre médicaments")
    plt.ylabel("Nombre patients")
    plt.yscale("log")
    plt.legend()
    fig.savefig(save_path)
    plt.show()
    plt.clf()


def get_proba_plot(df_analysis):
    filename = "Proba_analysis_" + args.mode +".pgf"
    save_path = os.path.join(args.save_dir, filename)

    ax = plt.axes()
    ax.plot(df_analysis.index, df_analysis.proba_nothealthy_nodrug,color="b", label="malformation sans prise du médicament")
    ax.plot(df_analysis.index, df_analysis.proba_nothealthy_drug,color="c", label="malformation avec prise du médicament")
    ax.set_xlabel('Médicaments par ordre décroissant de probabilité de malformation')
    ax.set_ylabel("Probablitité")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.legend()
    fig.savefig(save_path)
    plt.show()
    plt.clf()

def get_count_plot(df_analysis):
    filename = "count_" + args.mode +".pgf"
    save_path = os.path.join(args.save_dir, filename)

    plt.plot(df_analysis.index, df_analysis['count_diagnosis_0'],color="b", label="Nombre de grossesses saines",linewidth=0.5 )
    plt.plot(df_analysis.index, df_analysis['count_diagnosis_1'],color="c", label="Nombre de malformations",linewidth=0.5 )
    plt.xlabel("Médicaments par ordre décroissant de probabilité de malformation")
    plt.ylabel("Nombre patients")
    plt.yscale("log")
    plt.legend()
    fig.savefig(save_path)
    plt.show()
    plt.clf()
    

def get_combinaison(df):
    occ_df = pd.DataFrame(columns=['medicament','count_1','count_2','count_3','count_4','count_5',
                                   'count_6','count_7','count_8','count_9','count_10','count_11',
                                   'count_12','count_13','count_14','count_15','count_16',
                                   'count_17','count_18','count_19','count_20'])
    drugs = df.columns.tolist()
    dict= {}
    for drug in drugs:
        dict['medicament']=drug
        sum_row = df[df[drug] == 1].sum(axis=1).tolist()
        for i in range(1,21):
            dict['count_'+str(i)]= sum_row.count(i)
        occ_df = occ_df._append(dict, ignore_index=True) 
        dict={}  
    return occ_df

def get_map_comb_plot(df_cleaned, df_results):
    filename = "combinaison_subplots_" + args.mode +".pdf"
    save_path = os.path.join(args.save_dir, filename)

    occ_df = get_combinaison(df_cleaned.iloc[:,1:])
    merged_df = pd.merge(occ_df,df_results,on=['medicament'])
    merged_df = merged_df.sort_values(by=['proba_nothealthy_drug'],ascending=False)
    merged_df.reset_index(inplace=True)

    colormap = cm.get_cmap('viridis', 20)
    fig, ax = plt.subplots(5, 3, figsize=(15, 10))
    fig.set_size_inches((7,7.7))
    s=1
    for i in range(5):
        for j in range(3):
            ax[i,j].plot(merged_df.index,merged_df['count_'+str(s)],color=colormap(s), label=str(s)+' médicament')
            ax[i,j].legend()
            s+=1

    fig.supxlabel('Médicament')
    fig.supylabel('Fréquence de présence du médicament avec autres médicaments')
    plt.savefig(save_path)
    plt.show()
    plt.clf()

def get_NbCombinaison(df_cleaned):
    filename = "Histcombinaisons" + args.mode +".pdf"
    save_path = os.path.join(args.save_dir, filename)

    df_copy = df_cleaned.copy()
    df_copy.drop(columns=['diagnosis'],inplace=True)
    df_copy['count']=df_cleaned.sum(axis=1)
    df_copy_list = df_copy['count'].to_list()
    #df_copy_list = [x for x in df_copy_list]
    _ , bins, _ = plt.hist(df_copy_list, bins = len(df_copy['count'].value_counts()), histtype="bar",rwidth=0.7, color='lightseagreen')
    plt.xticks(((bins[:-1] + bins[1:])/2),np.arange(1,16))
    plt.xlabel("Nombre médicaments")
    plt.ylabel("Nombre patients")
    plt.yscale("log")
    fig.savefig(save_path)
    plt.show()
    plt.clf()


def main():

    #éviter répétition
    
    #read and clean data
    df = pd.read_csv("None_None\pharma_table.csv") #À changer 
    df = df[['ID', 'fin_grossesse','medicament','diagnosis']]
    print("Number of records before cleaning: ", len(df))
    df.drop_duplicates(inplace=True)
    print("Number of records after cleaning:", len(df))
    selected_columns = ['ID','fin_grossesse','medicament','diagnosis']
    sparse_df = pd.get_dummies(df[selected_columns], columns=['medicament'], sparse=True, prefix='', dtype=int)
    sparse_df = sparse_df.groupby(['ID','fin_grossesse','diagnosis']).sum().reset_index()
    print('Number of records after applying threshold:', len(sparse_df))
    sparse_df.to_csv("sparse_med_all.csv", index=False) 
    
    #TEMP FOR COMPATIBILITY
    with open("ordered_columns.json", "r") as file:
        desired_order = json.load(file)["columns"]
    sparse_df = sparse_df[desired_order]
    print("After Filtering: ")
    D, J = sparse_df.shape[0] , sparse_df.shape[1]-1
    print("Number of patients:", D)
    print("Number of medicament:", J)
    print("Cleaned data count classes:", sparse_df.diagnosis.value_counts())
    sparse_df.to_csv(os.path.join(args.save_data, 'sparse_med_cleaned.csv'),index=False)

    """
    df = pd.read_csv("sparse_med_all.csv")
    all_drugs = df.columns.to_list()
    all_drugs = all_drugs[3:]
    df = df[~(df[all_drugs] == 0).all(axis=1)]
    df.drop(columns=['ID','fin_grossesse'], inplace=True)

    print("Before Filtering: ")
    D, J = df.shape[0] , df.shape[1]-1
    print("Number of patients:", D)
    print("Number of medicament:", J)
    print("Original data count classes:", df.diagnosis.value_counts())

    df_analysis = get_analysis(df)
    df_analysis = df_analysis.sort_values(by=['proba_healthy_drug'],ascending=True)
    df_analysis.reset_index(drop=True, inplace=True)

    df_removed , df_results, df_end = get_filtered_meds(df_analysis)

    drugs_to_keep = df_results['medicament'].to_list()
    drugs_to_remove_start = df_removed['medicament'].to_list()
    drugs_to_remove_end = df_end['medicament'].to_list()
    

    dict_start , df_cleaned = remove_ID(drugs_to_remove_start, df) 
    print("Data shape after removing dangerous drugs under thres:", df_cleaned.shape) 
    print("Data count classes:", df_cleaned.diagnosis.value_counts())
    print("Details of each romeved drug:", dict_start)

    dict_end , df_cleaned = remove_ID(drugs_to_remove_end, df_cleaned) 
    print("Data shape after removing non dangerous:", df_cleaned.shape) 
    print("Data count classes:", df_cleaned.diagnosis.value_counts())
    print("Details of each romeved drug:", dict_end)
    
    columns_to_keep = ["diagnosis"] + drugs_to_keep
    df_cleaned = df_cleaned[columns_to_keep]
    

    print("After Filtering: ")
    D, J = df_cleaned.shape[0] , df_cleaned.shape[1]-1
    print("Number of patients:", D)
    print("Number of medicament:", J)
    print("Cleaned data count classes:", df_cleaned.diagnosis.value_counts())
    df_cleaned.to_csv(os.path.join(args.save_data, 'sparse_med_cleaned.csv'),index=False)
    """
    #Compte de nombre de médicament présent dans les combinaisons par classe 
    #get_comb_stats(df_results)
    
    #probilities plot
    #get_proba_plot(df_analysis)
    
    #get data count
    #get_count_plot(df_analysis)
    #Map des combinaisons dans lesquels les médicaments sont présents 
    #get_map_comb_plot(df_cleaned, df_results)

    # Histogramme de nombre de médicament par combinaison dans les records 
    #get_NbCombinaison(df_cleaned)

if __name__ == "__main__":
    main()



