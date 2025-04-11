import json

import pandas as pd
import numpy as np
import os
import csv

root_db = '/deepia/inutero/efemeris/data/EFEMERIS IRIT'
output_path = '/deepia/inutero/efemeris/data/efemeris_txt_v022025'

# Table paths
hospitmere_table_path = 'hospit_mere.csv'
interruption_table_path = 'interruption.csv'
malformation_table_path = 'malformation.csv'
naissance_table_path = 'naissance.csv'
pmsi_deces_table_path = 'pmsi_deces.csv'
thesaurus_icd10_table_path = 'thesaurus_icd10.csv'
thesaurus_pa_table_path = 'thesaurus_pa.csv'

### COLUMNS TO SEARCH
ordered_columns = ["_V04CX", "_A11CC03", "_C10AB02", "_N05AX13", "_L04AA06", "_N05AG02", "_C09BA02", "_V03AC03",
                   "_J05AF06", "_A10BB03", "_A10AD04", "_C09DA08", "_V01AA02", "_L04AA24", "_A06AD10", "_B03XA01",
                   "_A06AB02", "_J01CE08", "_H02AB10", "_C09DA01", "_N03AG04", "_C10AX09", "_M04AB01", "_C09AA08",
                   "_C02DB02", "_J05AE08", "_A10AE04", "_L04AA23", "_N04BA02", "_C03DB01", "_A10BH03", "_C09AA01",
                   "_M03BX02", "_C09CA08", "_N02AJ13", "_C09BA08", "_A07AA06", "_C09DA04", "_N05AX12", "_N03AA02",
                   "_N02AB03", "_C10AA01", "_H02AA02", "_A06AD11", "_M05BA07", "_N05AH01", "_H03AA02", "_J05AR03",
                   "_N05AE04", "_R01AC01", "_A06AG01", "_C07AA07", "_P02CC01", "_J01XB01", "_N04AA04", "_A10AB05",
                   "_N04AC01", "_J05AE03", "_C09CA01", "_C07AA03", "_L01BA01", "_A06AB06", "_J05AG01", "_L04AD02",
                   "_C10AB05", "_J05AR02", "_A10AE05", "_C09AA03", "_A11CC04", "_N06BA09", "_H04AA01", "_N05BA09",
                   "_N06BA12", "_A06AA02", "_N05AD01", "_C09AA10", "_L02AE03", "_C09BA03", "_C01BA03", "_V03AF03",
                   "_J05AX08", "_J04AK02", "_C07AG02", "_C07AB04", "_N03AF02", "_J05AR06", "_C10AA02", "_C07AA12",
                   "_C01EB03", "_J04BA02", "_A03AX04", "_R03AL02", "_J01DH02", "_C10AA07", "_A06AD15", "_C03CA01",
                   "_J04AC01", "_C01DA02", "_N04BC01", "_N06BA02", "_C10AC01", "_C09CA03", "_N07CA03", "_R03BA07",
                   "_B01AC06", "_A02BC06", "_L04AX01", "_N03AX11", "_N06BA04", "_A10AD01", "_A09AA04", "_H01CC02",
                   "_A10AB04", "_A02BC05", "_A11CA01", "_C09BA05", "_C07AB07", "_C05AA51", "_N06AX21", "_J05AR04",
                   "_C09AA05", "_L02AE02", "_N07BA03", "_B01AA03", "_N03AX12", "_A10BB01", "_N07CA01", "_H01CC01",
                   "_J05AE10", "_C07AG01", "_C08CA01", "_N06AX11", "_H03BB02", "_C01CA24", "_N02BA01", "_B01AB04",
                   "_C08DA01", "_C03AA03", "_P01BB51", "_N05AA02", "_N03AX09", "_A10BA02", "_A10AC01", "_L02AE01",
                   "_N03AD01", "_N07BC51", "_C02CA01", "_N03AX16", "_J05AH01", "_A10AB06", "_H01CA02", "_A04AD11",
                   "_N03AA03", "_B05ZA", "_A10BF01", "_R03DX05", "_C09BA04", "_A10BB09", "_M01AC02", "_N07AA02",
                   "_R06AA02", "_N03AG01", "_H02AB04", "_N05AH04", "_R03DC03", "_A03FA01", "_C02AB02", "_C10AA05",
                   "_A07EC01", "_L01BB02", "_C08CA05", "_A12BA01", "_A11CC01", "_B01AB10", "_A12AA04", "_M01AB01",
                   "_H02AB01", "_A10BG02", "_R03AC13", "_C07AB02", "_R03AK07", "_N03AF01", "_N06AB04", "_A02BD07",
                   "_C09CA06", "_N02AA01", "_A11CC05", "_N02CC03", "_N06AX16", "_L01XX05", "_M03AX01", "_H02AB09",
                   "_N06BA01", "_N05AN01", "_R01AD12", "_R05DA04", "_R01AD05", "_A02BC04", "_R03AK06", "_B02AA02",
                   "_A10AB01", "_P01BC01", "_J05AB11", "_J01EA01", "_N02AA03", "_L01BC02", "_N05AH02", "_N05AB06",
                   "_N05AX08", "_N06AX05", "_N05AH03", "_N02AA05", "_R03AC12", "_R03BA08", "_B03BB01", "_A11JA",
                   "_C09CA04", "_N05BA06", "_N02CC05", "_J01DD08", "_J01DC10", "_C03EA01", "_L04AB04", "_A03FA03",
                   "_H02AB02", "_A12AX", "_H02AB06", "_A07DA03", "_R03AK04", "_R03BA02", "_N07BA01", "_L02BG04",
                   "_R01AD09", "_N05AB04", "_N04AA01", "_N06AA04", "_A05AA02", "_C10AC02", "_J01DB04", "_L04AB02",
                   "_C01CA17", "_N05AF01", "_N03AB02", "_N06AB10", "_N06AA01", "_A02BC02", "_N02BE01", "_M01AE02",
                   "_B03BA01", "_A02BA02", "_R03AC03", "_J01MA14", "_A11HA02", "_N02CA01", "_J01MA16", "_C07AA05",
                   "_J01DD04", "_A02BC03", "_P01BA02", "_N06AA09", "_H02AB07", "_C09AA09", "_J05AB01", "_J01FA15",
                   "_H03AA01", "_N06AA10", "_B03AA07", "_P02CA01", "_M03BX08", "_J01CR02", "_M03BC01", "_J01DB01",
                   "_C09AA04", "_C09CA07", "_C03BA11", "_R03AC02", "_J02AC01", "_J01FF01", "_J01AA02", "_N02AB02",
                   "_N06AB06", "_N06AX12", "_N03AE01", "_R03BA05", "_J01XE01", "_C03DA01", "_A07CA", "_P01AB01",
                   "_N05BA04", "_C09DA07", "_N05AB02", "_N02CC06", "_C09DA03", "_M05BA04", "_N05BB01", "_N05CD02",
                   "_A03FA02", "_J01AA08", "_J01MA02", "_H03BA02", "_A07AA02", "_R03BB04", "_C02AC01", "_N03AX14",
                   "_N06AB03", "_N05BA12", "_A02BB01", "_J02AB02", "_N07BC", "_N02CC02", "_C09AA02", "_M01AH01",
                   "_M01AE01", "_J01FA10", "_N05BA02", "_N06AA12", "_R06AA59", "_N02AF01", "_B01AB05", "_J01XA01",
                   "_A11BA", "_R03AC08", "_C05BA04", "_A12BA02", "_J01FA09", "_N02CC04", "_J05AH02", "_A02BC01",
                   "_M01AB05", "_J01MA01", "_C08DB01", "_J01CA04", "_N02AJ06", "_C05AA01", "_J05AR01", "_J05AR10",
                   "_M01AC06", "_J01DB05", "_N06AB05", "_L03AB08", "_J01EE02", "_N05AF05", "_M01AC01", "_P01BA01",
                   "_C07AB03", "_J01CF02", "_R01AD11", "_N05BA01", "_J01CE02", "_N05BA08", "_J01FA02", "_J05AB09",
                   "_N05CD07", "_N01BB02", "_N02CC01", "_H05BA01", "_M03BX01", "_M04AC01", "_C01AA05", "_M01AB55",
                   "_R01AD08", "_A01AC01", "_N05BE01", "_M01AE03", "_J01GB01", "_M01AG01", "_J01DC02", "_J01EE01",
                   "_M01AH02", "_A09AA02", "_A04AA01", "_B03AC", "_N05CD01", "_N06AA02", "_H02AB08", "_J01XD01",
                   "_R03BB01", "_R03BA01", "_M01AE09", "_A10BG03", "_J05AF02", "_A11DA01", "_J01MA12", "_J01CE10",
                   "_L03AB07", "_A07EA02", "_J04AB02", "_J01FA01", "_A07EC02", "_P01BC02", "_J01CA02", "_R03AC04",
                   "_C09AA06", "_A07AA09", "_H02BX01", "_R03DA04", "_J05AE04", "_A02BX02", "_R01AD01", "_R03CC02",
                   "_N06AB08", "_A02BA03", "_N06AX02", "_J02AC02", "_N06AX06", "_H01BA02", "_A02BA01", "_N05AA01",
                   "_C10AA03", "_R03CB03", "_J01AA07", "_J01MA06", "_J05AG03", "_M01AX01", "_L04AD01", "_N02CX01",
                   "_B01AB01", "_J05AE01", "_J01RA02", "_J01CA01", "_L02BA01", "_J05AE02", "_N05AB03", "_J01DC04",
                   "_R01AX03", "_N04BB01", "_N06AA06", "_J05AF05", "_M01AB08", "_A02BA04", "_A07DA01", "_M01AE11",
                   "_N02BA11", "_R06AD02", "_A10AE01", "_N04BC05", "_J01CA06", "_N02AD01", "_R03DC01", "risk",
                   "diagnosis"]

def read_csv(filename: str, dtypes: dict = {}) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=';', dtype=dtypes, encoding="ISO-8859-1", decimal=",")
    return df

def read_csv_prescription(db_path: str, lang: str = 'en') -> pd.DataFrame:
    """
    Read the PRESCRIPTIONS table
    We consider all columns as pertinent and use NUM_GROSSESSE to align with ISSUE (main)

    Index(['NUM_GROSSESSE', 'DATE_DEL', 'DELAI_PRESCRIPTION_SA',
           'PRESC_GROSSESSE_SA', 'ORGANO', 'TRIM_PRESCRIPTION_SA', 'NOM',
                  'SPECIALITE', 'CIP', 'atc', 'FORME', 'SPE_MEDECIN', 'QUANT', 'DATE_LMP',
                         'LIBELLE_FORME', 'LIBELLE_SPECIALITE']
    """
    db_name = 'prescription.csv'
    atc_name = 'thesaurus_pa.csv'
    if lang == 'en':
        db_name = 'prescription_en.csv'
        atc_name = 'thesaurus_pa_en.csv'
    dtypes = {'NUM_GROSSESSE': str, 'DATE_DEL': str, 'DELAI_PRESCRIPTION_SA': 'Int64',  # nb days ,
              'PRESC_GROSSESSE_SA': 'Int64', 'TRIM_PRESCRIPTION_SA': 'Int64', 'NOM': str, 'SPECIALITE': str,
              'CIP': str, 'atc': str, 'FORME': 'Int64', 'SPE_MEDECIN': 'Int64', 'QUANT': 'Int64', 'DATE_LMP': str,
              'LIBELLE_FORME': str, 'LIBELLE_SPECIALITE': str}
    df = read_csv(os.path.join(db_path, db_name), dtypes=dtypes)
    df = df.set_index('NUM_GROSSESSE')
    for col in ['DATE_LMP', 'DATE_DEL']:
        df[col] = df[col].astype('datetime64[ns]')

    # ADD ATC INFO
    # Create a dictionary with lists for multiple values in col1
    df_atc = pd.read_csv(os.path.join(db_path, atc_name), delimiter=';', encoding="ISO-8859-1")
    atc_dict = df_atc.groupby('SPECIALITE')['ATCLIB'].apply(list).to_dict()

    def map_with_check(atc):
        if atc in atc_dict:
            if atc == 'TOCTINO' or atc == 'CONTRACNE':
                return ''
            try:
                return ', '.join(atc_dict[atc])
            except:
                print(atc, atc_dict[atc])
        else:
            print(f"Warning: Key '{atc}' not found in dictionary.")
            return None  # Return None or any default value if key not found

    # Map the values and create a new column 'Medications'
    df['LIBELLE_ATC'] = df['SPECIALITE'].apply(map_with_check)
    return df


if __name__ == '__main__':
    df_prescriptions = read_csv_prescription(root_db, lang='fr')
    df_prescriptions = df_prescriptions[['atc', 'LIBELLE_ATC', 'TRIM_PRESCRIPTION_SA']]
    df_prescriptions = df_prescriptions[df_prescriptions['TRIM_PRESCRIPTION_SA']<=1]
    df_prescriptions.drop(columns=['TRIM_PRESCRIPTION_SA', 'LIBELLE_ATC'], inplace=True)
    print(df_prescriptions.shape)
    print(df_prescriptions.head())

    one_hot_df = pd.crosstab(df_prescriptions.index, df_prescriptions['atc'])
    print(one_hot_df.shape)
    print(one_hot_df.head())
    efemeris_columns = set("_"+x for x in list(one_hot_df.columns))
    montreal_columns = set(ordered_columns[:-2])

    print(f"In total {len(efemeris_columns)} in EFEMERIS and {len(montreal_columns)} in MONTREAL")
    print(f"with {len(efemeris_columns.intersection(montreal_columns))} columns in common")
    print(f"{len(efemeris_columns.difference(montreal_columns))} ATC codes from EFEMERIS not in MONTREAL")
    print(f"{len(montreal_columns.difference(efemeris_columns))} ATC codes from MONTREAL not in EFEMERIS")

    # Reorder columns to follow the order in 'ordered_columns'
    common_columns = efemeris_columns.intersection(montreal_columns)
    ordered_columns = [c[1:] if c[0]=='_' else c for c in ordered_columns ]

    one_hot_df_filled_0 = one_hot_df.reindex(columns=ordered_columns, fill_value=0)
    one_hot_df_filled_nan = one_hot_df.reindex(columns=ordered_columns)

    print("Filled 0 ")
    print(one_hot_df_filled_0.shape)
    print(one_hot_df_filled_0.head())

    print("Filled NaN ")
    print(one_hot_df_filled_nan.shape)
    print(one_hot_df_filled_nan.head())

    # Load train/dev/test and assign y (diagnosis)
    one_hot_df_filled_0.drop(columns='diagnosis', inplace=True)
    for split in ['train', 'dev', 'test']:
        ifile = os.path.join(output_path, 'all', f'en_random_{split}', 'txt_presc_t1.csv')
        df = pd.read_csv(ifile, delimiter=';')
        df = df.set_index('NUM_GROSSESSE')
        df = df[['MALFO_MAJ']].rename(columns={'MALFO_MAJ': 'diagnosis'})
        print("Before join", df.shape)
        final_df = df.join(one_hot_df_filled_0, how='left')
        print("After join", final_df.shape)
        final_df = final_df.reindex(columns=ordered_columns)
        final_df.to_csv(os.path.join(output_path, f'all_random_montreal_meds_sparse_{split}.csv'), index=False)
        print("---")

    # Approach using columns present in both datasets, and keeping patients with these meds
    print("Working with Common approach")
    common_columns = [c[1:] if c[0]=='_' else c for c in common_columns ]
    # keep order
    filtered_ordered_columns = [ c for c in ordered_columns if c in common_columns]
    filtered_df = df_prescriptions[df_prescriptions['atc'].isin(filtered_ordered_columns)]

    one_hot_df_common = pd.crosstab(filtered_df.index, filtered_df['atc'])
    print(one_hot_df_common.shape)

    one_hot_df_filled_common = one_hot_df_common[filtered_ordered_columns].reindex(columns=filtered_ordered_columns)
    print(one_hot_df_common.shape)
    print(one_hot_df_filled_common.head())

    # Load train/dev/test and assign y (diagnosis), for only common meds
    filtered_ordered_columns.extend(['risk', 'diagnosis'])
    for split in ['train', 'dev', 'test']:
        ifile = os.path.join(output_path, 'all', f'en_random_{split}', 'txt_presc_t1.csv')
        df = pd.read_csv(ifile, delimiter=';')
        df = df.set_index('NUM_GROSSESSE')
        df = df[['MALFO_MAJ']].rename(columns={'MALFO_MAJ': 'diagnosis'})
        print("Before join", df.shape)
        final_df = df.join(one_hot_df_filled_common, how='inner')
        print("After join", final_df.shape)
        final_df = final_df.reindex(columns=filtered_ordered_columns)
        final_df.to_csv(os.path.join(output_path, f'all_random_montreal_common_meds_sparse_{split}.csv'), index=False)
        print("---")

    with open('common_ordered_columns.json', 'w') as f:
        forfile = {'columns': filtered_ordered_columns}
        json.dump(forfile, f, ensure_ascii=False, indent=4)


"""
In total 1096 in EFEMERIS and 439 in MONTREAL
with 350 columns)
746 from EFEMERIS not in MONTREAL
89 from MONTREAL not in EFEMERIS


with common approach
Train
Before join (95413, 1)
After join (74036, 351)
---
Dev
Before join (31804, 1)
After join (24784, 351)
---
Test
Before join (31805, 1)
After join (24599, 351)

"""