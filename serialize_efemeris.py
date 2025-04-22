"""
Script to create an extended and filtered version of EFEMERIS.
We filter out multiple pregnancies
We filter out hospitalizations related to malformations
"""

import pandas as pd
import numpy as np
import os
import csv
from tqdm import tqdm
from collections import Counter

root_db = /path/to/db
output_path = output/path

# Table paths
hospitmere_table_path = 'hospit_mere.csv'
interruption_table_path = 'interruption.csv'
malformation_table_path = 'malformation.csv'
naissance_table_path = 'naissance.csv'
pmsi_deces_table_path = 'pmsi_deces.csv'
thesaurus_icd10_table_path = 'thesaurus_icd10.csv'
thesaurus_pa_table_path = 'thesaurus_pa.csv'

"""
12-12-2024
We are only working with main_table,precarite, and prescription table
The goal is to filter relevant columns, adapt them in numeric and text formats.
"""


def read_csv(filename: str, dtypes: dict = {}) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=';', dtype=dtypes, encoding="ISO-8859-1", decimal=",")
    return df


def read_csv_main(db_path: str) -> pd.DataFrame:
    """
    Read the Issue table from EFEMERIS (main), and filter only the columns:
    GROSSESSE_MULTIPLE: (0/1),  SEXE: F/M, AGE: numéro (de la mère),
    ALD:  0/1 - affection de longue durée, DATE_LMP - date des dernières
    règles ( < début de grossese),, DEB_GROSSESSE - date, ECHO - number,
    HOSPIT - number, CAT_AGE, ALCOOL, TABAC, GESTITE, HTA_TOT, DIABETE_PRESC,
    ATCD_DIABETE, DIAB_GESTA, and MALFO_MAJEURE (y to predict)
    """
    db_name = 'issue.csv'
    selected_columns_to_align = ['ID', 'ID_MARITAL', 'ID_PATRO', 'IMG_NUM', 'dossier_PMI',
                                 'BASE_CPAM', 'NUM_MERE', 'MEDOC_MOINS3M', 'NUM_GROSSESSE', 'CERTIFICAT8J',
                                 'CERTIFICAT9M',
                                 'CERTIFICAT24M', 'PARITE', 'ISSUE_EUROCAT']
    selected_columns_to_align = ['ID', 'NUM_MERE', 'NUM_GROSSESSE']  # Simplified version by now

    selected_columns = ['GROSSESSE_MULTIPLE', 'SEXE', 'AGE', 'ALD', 'DATE_LMP', 'DEB_GROSSESSEBIS',
                        'DEB_GROSSESSE', 'ECHO', 'HOSPIT', 'CAT_AGE', 'MALFO_MAJEURE', 'ALCOOL', 'TABAC',
                        'ATCD_DIABETE',
                        'DIAB_GESTA', 'GESTITE', 'HTA_TOT', 'DIABETE_PRESC']
    columns_to_select = selected_columns + selected_columns_to_align

    dtypes = {'GROSSESSE_MULTIPLE': 'Int64', 'SEXE': str, 'AGE': 'Int64', 'ALD': 'Int64', 'ECHO': 'Int64',
              'HOSPIT': 'Int64', 'CAT_AGE': str, 'MALFO_MAJEURE': 'Int64', 'ALCOOL': 'Int64', 'TABAC': 'Int64',
              'ATCD_DIABETE': 'Int64', 'DIAB_GESTA': 'Int64', 'GESTITE': 'Int64', 'HTA_TOT': 'Int64',
              'DIABETE_PRESC': 'Int64', 'ID': 'Int64', 'NUM_MERE': 'Int64', 'NUM_GROSSESSE': str}

    df = read_csv(os.path.join(db_path, db_name), dtypes=dtypes)

    for col in ['DATE_LMP', 'DEB_GROSSESSEBIS', 'DEB_GROSSESSE']:
        df[col] = df[col].astype('datetime64[ns]')

    df['ALCOOL'] = df['ALCOOL'].fillna(0)
    df['TABAC'] = df['TABAC'].fillna(0)
    df = df[columns_to_select]
    # df = df.set_index('ID')
    return df


def read_csv_precarite(db_path: str) -> pd.DataFrame:
    """
    Read the INDIC_GEO_PRECARITE table
    We consider all columns as pertinent and use NUM_GROSSESSE to align with ISSUE (main)
    """
    db_name = 'indic_geo_precarite.csv'
    dtypes = {'NUM_GROSSESSE': str, 'POP': str, 'DENSITE_GEO': 'Int64', 'FDEP': float, 'CMUACS': 'Int64'}

    df = read_csv(os.path.join(db_path, db_name), dtypes=dtypes)
    df = df.set_index('NUM_GROSSESSE')

    # Handle 3 cases which are duplicates
    df = df.groupby(df.index).last()
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


def read_csv_hospitalization(db_path: str, lang: str = 'en') -> pd.DataFrame:
    """
    Read the Hospitalizations table
    We consider all columns as pertinent and use NUM_GROSSESSE to align with ISSUE (main)
    """
    db_name = 'hospit_mere.csv'
    if lang == 'en':
        db_name = 'hospit_mere_en.csv'

    dtypes = {'PMSI_HOSP_DIAG': str, 'PMSI_TYPE_DIAG': 'Int64', 'PMSI_HOSP_DtEntree': str,
              'PMSI_HOSP_Dtsortie': str, 'DATE_ACC': str, 'PMSI_HOSP_Libelle_Diag': str, 'NUM_GROSSESSE': str,
              'NUM_MERE': str}

    df = read_csv(os.path.join(db_path, db_name), dtypes=dtypes)
    df = df.set_index('NUM_GROSSESSE')

    for col in ['PMSI_HOSP_DtEntree', 'PMSI_HOSP_Dtsortie', 'DATE_ACC']:
        df[col] = df[col].astype('datetime64[ns]')

    df.drop(columns=['DATE_ACC'], inplace=True)
    return df


def prepare_profile_information(db_path: str):
    """
    Generates numeric and text based datasets for the profile
    and geo_precarite tables.

    The generated files are specified in the Readme.
    """
    df_issue = read_csv_main(db_path)
    # Remove multiple pregnancy
    df_issue = df_issue[df_issue['GROSSESSE_MULTIPLE'] == 0].drop(columns=['GROSSESSE_MULTIPLE'])
    df_precarite = read_csv_precarite(db_path)

    # Put everything in one table
    df_profile = df_issue.merge(df_precarite, on='NUM_GROSSESSE', how='left')
    print(f"Shapes for Issue {df_issue.shape}, for Precarite {df_precarite.shape}, and merged {df_profile.shape}")

    # Merge columns DEB_GROSSESSE and DEB_GROSSESSEBIS if the first is Nan
    df_profile['DEB_GROSSESSE'] = df_profile['DEB_GROSSESSE'].fillna(df_profile['DEB_GROSSESSEBIS'])
    total_days = df_profile['DEB_GROSSESSE'] - df_profile['DATE_LMP']
    # Compute days between debut grossesse and date LMP
    df_profile['NB_JOURS_DEB'] = [max(0, x.days) for x in total_days]

    # Prepare text-based version (using templates)
    densite_values = {
        1: "very dense",
        2: "dense",
        3: "not very dense",
        4: "very sparse",}
    densite_values_fr = {
        1: "très dense",
        2: "dense",
        3: "peu dense",
        4: "très peu dense", }

    col_to_str = {
        'SEXE': lambda x: f"The fœtus is {'female' if x == 'F' else 'male' if x == 'M' else 'of unknown gender'}",
        #'ECHO': lambda x: f"and exposed to {int(x)} ultrasounds." if (
        #    not pd.isna(x)) else "and without exposition to ultrasounds.",
        #'HOSPIT': lambda x: f"The fœtus was hospitalized {x} days in the following 3 months after birth."
        #    if ( not pd.isna(x)) and (int(x) == 1) else "The fœtus was not hospitalized.",
        'AGE': lambda x: f"The patient is {int(x)} years old",
        'DIAB_GESTA': lambda x: "The patient has gestational diabetes" if (not pd.isna(x)) and (
                int(x) == 1) else "The patient does not have gestational diabetes",
        'ALD': lambda x: "The patient has a long-term illness" if (not pd.isna(x)) and (
                int(x) == 1) else "The patient does not have a long-term illness" if (not pd.isna(x)) and (
                int(x) == 0) else "No long-term illness known",
        'ATCD_DIABETE': lambda x: "The patient has a history of diabetes" if (not pd.isna(x)) and (
                int(x) == 1) else "The patient does not have a history of diabetes",
        'ALCOOL': lambda x: "The patient declares she drinks alcohol" if int(x) == 1 else "The patient declares she does not drink alcohol",
        'TABAC': lambda x: "The patient declares she smokes" if int(x) == 1 else "The patient declares she does not smoke",
        'HTA_TOT': lambda x: "The patient has hypertension" if (not pd.isna(x)) and (
                int(x) == 1) else "The patient does not have hypertension",
        'GESTITE': lambda x: f"This is the patient's {x}{'st' if x == 1 else 'nd' if x == 2 else 'rd' if x == 3 else 'th'} pregnancy" if (not pd.isna(x)) and (int(x) > 0) else "",
        'DIABETE_PRESC': lambda x: "The patient is prescribed  diabetes medication" if (not pd.isna(x)) and (
                    int(x) == 1) else "The patient is not prescribed diabetes medication",
        'CMUACS': lambda x: "The patient is covered by CMU/ACS health insurance" if (not pd.isna(x)) and (
                int(x) == 1) else "The patient is not covered by CMU/ACS health insurance",
        'FDEP': lambda x: f"The socio-economic index (FDEP) is {x}" if str(x) != 'nan' else "",
        'DENSITE_GEO': lambda x: f"The patient lives in an area classified as {densite_values[x]}" if pd.notna(x) else "",
        'POP': lambda x: f'The patient lives in an area with a population density between '
                         f'{x.replace("a ","and ").replace("et plus", "and more").replace("de", "").replace("à", "and")} inhabitants per square kilometer'
        if (str(x) != 'nan' and pd.notna(x)) else "",

        # 'CAT_AGE': lambda x: f"The patient age is categorized as {x}.",
        'GROSSESSE_MULTIPLE': lambda x: "Multiple pregnancy" if (not pd.isna(x)) and (
                int(x) == 1) else "Single pregnancy",
        'NB_JOURS_DEB': lambda x: f"The pregnancy started {x} days after the last period",
        # 'ID': lambda x: f"The patient ID is {x}.",
        # 'NUM_MERE': lambda x: f"The mother’s identifier is {x}.",
        # 'NUM_GROSSESSE': lambda x: f"The pregnancy number is {x}.",
        'WEEKS_DEB_GROSS': lambda
            x: f"There are {x} weeks between date of last menstrual period and start of pregnancy",
        # 'MALFO_MAJEURE': lambda x: "There was a major malformation." if x == 1 else "There was no major malformation."
    }
    # Prepare text-based version (using templates)
    col_to_str_fr = {
        'SEXE': lambda x: f"Le fœtus est {'féminin' if x == 'F' else 'masculin' if x == 'M' else 'de genre inconnu'}",
        #'ECHO': lambda x: f"et exposé à {int(x)} échographies." if (
        #    not pd.isna(x)) else "et sans exposition aux échographies.",
        #'HOSPIT': lambda x: f"Le fœtus a été hospitalisé {x} jours dans les 3 mois suivant la naissance."
        #    if (not pd.isna(x)) and (int(x) == 1) else "Le fœtus n'a pas été hospitalisé",
        'AGE': lambda x: f"La patiente a {int(x)} ans",
        'DIAB_GESTA': lambda x: "La patiente a un diabète gestationnel" if (not pd.isna(x)) and (
                int(x) == 1) else "La patiente n'a pas de diabète gestationnel",
        'ALD': lambda x: "La patiente a une affection de longue durée" if (not pd.isna(x)) and (
                    int(x) == 1) else "La patiente n'a pas d'affection de longue durée" if (not pd.isna(x)) and (
                    int(x) == 0) else "Aucune affection de longue durée connue",
        'ATCD_DIABETE': lambda x: "La patiente as des antécédents de diabète" if (not pd.isna(x)) and (
                    int(x) == 1) else "La patient n'as pas des antécédents de diabète",
        'ALCOOL': lambda x: "La patiente déclare qu'elle boit d'alcool" if int(
            x) == 1 else "la patiente déclare qu'elle ne boit pas d'alcool",
        'TABAC': lambda x: "la patiente déclare qu'elle fume" if int(x) == 1 else "la patiente déclare qu'elle ne fume pas",
        'HTA_TOT': lambda x: "La patiente est hypertendue" if (not pd.isna(x)) and (
                int(x) == 1) else "La patiente n'est pas hypertendue",
        'GESTITE': lambda x: f"C'est la {x}ᵉ grossesse de la patiente" if (not pd.isna(x)) and (int(x) > 0) else "",
        'DIABETE_PRESC': lambda x: "La patiente est sous prescription de médicaments contre le diabète"
            if (not pd.isna(x)) and (int(x) == 1) else "La patiente n'est pas sous prescription de médicaments contre le diabète",
        'CMUACS': lambda x: "La patiente bénéficie d'une aide à la couverture santé (CMU/ACS)" if (not pd.isna(x)) and (
                int(x) == 1) else "La patiente ne bénéficie pas d'une aide à la couverture santé (CMU/ACS)",
        'FDEP': lambda x: f"L'indice socio-économique (FDEP) est de {x}" if str(x) != 'nan' else "",
        'DENSITE_GEO': lambda x: f"La zone est classée au niveau {densite_values_fr[x]} de densité géographique" if pd.notna(x) else "",
        'POP': lambda x: f"Le patient vit dans une région dont la densité de population est comprise {x}" if str(x) != 'nan' else "",


        # 'CAT_AGE': lambda x: f"The patient age is categorized as {x}.",
        'NB_JOURS_DEB': lambda x: f"La grossesse a débuté {x} jours après les dernières règles",
        # 'ID': lambda x: f"The patient ID is {x}.",
        # 'NUM_MERE': lambda x: f"The mother’s identifier is {x}.",
        # 'NUM_GROSSESSE': lambda x: f"The pregnancy number is {x}.",
        'GROSSESSE_MULTIPLE': lambda x: "Grossesse multiple" if (not pd.isna(x)) and (
                int(x) == 1) else "Grossesse simple",
        'WEEKS_DEB_GROSS': lambda
            x: f"Il y a {x} semaines entre la date des dernières règles et le début de la grossesse",
        # 'MALFO_MAJEURE': lambda x: "There was a major malformation." if x == 1 else "There was no major malformation.",
    }

    build_paragraphs_geo = ['SEXE', 'AGE', 'DIAB_GESTA', 'ALD', 'ATCD_DIABETE', 'ALCOOL', 'TABAC', 'HTA_TOT', 'GESTITE',
                            'CMUACS', 'FDEP', 'DENSITE_GEO', 'POP']

    # Create txt with only profile information
    for lang_template, lang_str in zip([col_to_str, col_to_str_fr], ['en', 'fr']):
        print("Language", lang_str)
        # Create txt with profile + geoprecarite information

        text_df = {'MALFO_MAJEURE': df_profile['MALFO_MAJEURE'], 'ID': df_profile['ID'],
                   'NUM_MERE': df_profile['NUM_MERE'], 'NUM_GROSSESSE': df_profile['NUM_GROSSESSE'],
                   'txt': df_profile.apply(
                       lambda x: ". ".join([(lang_template[col])(x[col]) for col in build_paragraphs_geo
                                            if len((lang_template[col])(x[col])) > 0]) +".", axis=1)}
        new_df = pd.DataFrame(text_df)
        filename = f'txt_profilegeo_{lang_str}.csv'
        print(f"Save text version with profile information and geo information into {filename}")
        new_df.to_csv(os.path.join(output_path, filename), sep=';')#, quoting = csv.QUOTE_ALL, escapechar = "\\")


def prepare_prescription_information_txt(db_path: str):
    """
    Generate text version of the prescription table, filtering by dates
    and fine-grained content
    """
    col_to_str_meds = {
        'NOM': lambda x: f"- {x}" if pd.notna(x) else "",
        'CIP': lambda x: f"CIP {x.split('.')[0]}" if pd.notna(x) else "",
        'LIBELLE_FORME': lambda x: f"As {x.lower()}" if pd.notna(x) else "",
        #'FORME': lambda x: f"({x})" if pd.notna(x) else "",
        'SPECIALITE': lambda x: f"The specialty is {x.lower().title()}" if pd.notna(x) else "",
        'LIBELLE_ATC': lambda x: f"The active principle(s) are {x}" if pd.notna(x) else "",
        'atc': lambda x: f"ATC {x}" if pd.notna(x) else "",
        'QUANT': lambda x: f"{x} boxes prescribed" if pd.notna(x) else "At least 1 box was prescribed",
        'DELAI_PRESCRIPTION_SA': lambda x: f"The dispensation was at the {x} th day" if pd.notna(x) else "",
        'PRESC_GROSSESSE_SA': lambda
            x: f"The dispensation was made during pregnancy" if x == 1 else "The dispensation was made before pregnancy",
        'TRIM_PRESCRIPTION_SA': lambda x: f"In trimester {x}" if x > 0 else "",
        'LIBELLE_SPECIALITE': lambda x: f"Prescribed by the {x}" if pd.notna(x) else "",
        # 'WEEKS_DEL': lambda x: f", delivered at {x} weeks" if pd.notna(x) else "",
        # 'ORGANO': lambda x: ", during organogenesis," if x == 1 else "",
        # 'SPE_MEDECIN': lambda x: f"prescribed by {x}" if pd.notna(x) else "",
    }
    col_to_str_meds_fr = {
        'NOM': lambda x: f"- {x}" if pd.notna(x) else "",
        'CIP': lambda x: f"CIP {x.split('.')[0]}" if pd.notna(x) else "",
        'LIBELLE_FORME': lambda x: f"Comme {x.lower()}" if pd.notna(x) else "",
        'SPECIALITE': lambda x: f"La specialité est {x.lower().title()}" if pd.notna(x) else "",
        'LIBELLE_ATC': lambda x: f"Le(s) principes active(s) {x}" if pd.notna(x) else "",
        'atc': lambda x: f"ATC {x}" if pd.notna(x) else "",
        'QUANT': lambda x: f"{x} boîtes fournies" if pd.notna(x) else "Au moins 1 boîte fournie.",
        'DELAI_PRESCRIPTION_SA': lambda x: f"La dispensation a eu lieu au {x}ᵉ jour" if pd.notna(x) else "",
        'PRESC_GROSSESSE_SA': lambda
            x: f". La prescription a été faite pendant la grossesse" if x == 1 else ". La prescription a été faite avant la grossesse",
        'LIBELLE_SPECIALITE': lambda x: f"Prescrite par le {x}" if pd.notna(x) else "",
        'FORME': lambda x: f"({x})" if pd.notna(x) else "",
        'TRIM_PRESCRIPTION_SA': lambda x: f", au {x}ᵉ trimestre" if x > 0 else ", avant la grossesse",

        # 'WEEKS_DEL': lambda x: f", delivered at {x} weeks" if pd.notna(x) else "",
        # 'ORGANO': lambda x: ", during organogenesis," if x == 1 else "",
        # 'SPE_MEDECIN': lambda x: f"prescribed by {x}" if pd.notna(x) else "",
    }

    build_paragraph = ['NOM', 'CIP', 'LIBELLE_FORME', 'SPECIALITE',  'LIBELLE_ATC', 'atc', 'QUANT',
                 'DELAI_PRESCRIPTION_SA', 'PRESC_GROSSESSE_SA', 'TRIM_PRESCRIPTION_SA', 'LIBELLE_SPECIALITE']

    for lang_template, lang_str in zip([col_to_str_meds, col_to_str_meds_fr], ['en', 'fr']):
        print("Language", lang_str)
        df_prescription = read_csv_prescription(db_path, lang=lang_str)
        df_prescription = df_prescription.drop(columns=['DATE_LMP', 'DATE_DEL', 'ORGANO'])
        print(len(set(df_prescription.index.values)))

        for trim in [-1, 1, 2, 3]:
            # Filterout trimester dataset
            if trim == -1:
                tmp_df = df_prescription[(df_prescription['DELAI_PRESCRIPTION_SA'] >= -31)
                                         & (df_prescription['DELAI_PRESCRIPTION_SA'] <= 97)]
            else:
                tmp_df = df_prescription[df_prescription['TRIM_PRESCRIPTION_SA'] <= trim]

            text_df = {}
            text_df['NUM_GROSSESSE'] = tmp_df.index.values
            text_df['DELAI_PRESCRIPTION_SA'] = tmp_df['DELAI_PRESCRIPTION_SA'].values
            text_df['txt'] = tmp_df.apply(
                lambda x: ". ".join([(lang_template[col])(x[col]) for col in build_paragraph
                                    if len((lang_template[col])(x[col])) > 0]) +".", axis=1)
            # Add num ID issue, num ere, num grossesse into jsonl
            new_df = pd.DataFrame(text_df)
            new_df.reset_index(drop=True, inplace=True)
            # First sort by prescription date, then group and concatenate by num_grossesse
            final_df = new_df.sort_values('DELAI_PRESCRIPTION_SA').groupby('NUM_GROSSESSE').txt.agg('\n'.join).reset_index()
            print("Final dataset shape", final_df.shape)

            filename = f'cleaned_txt_prescriptions_t{trim}_{lang_str}.csv'
            if trim == -1:
                filename = f'cleaned_txt_prescriptions_t_window_{lang_str}.csv'
            print("Save generated data into ", filename)
            final_df.to_csv(os.path.join(output_path,  filename), sep=';')#, quoting = csv.QUOTE_ALL, escapechar = "\\")


def prepare_hosp_information_txt(db_path: str):
    """
    Generate text version of the hospitalization table, filtering by dates
    and fine-grained content
    """
    col_to_str = {
        # Hospit_mere
        'DELAI_HOSP': lambda x: f'- Hospital entry at {x}th pregnancy day',
        'DAYS_HOSP': lambda x: f'Hospital stay during {x} days',
        'PMSI_TYPE_DIAG': lambda x: '- Main diagnostic' if int(x) == 1 else '- Associated diagnosis' if int(
            x) == 2 else '- Related diagnosis',
        'PMSI_HOSP_DIAG': lambda x: f'CIM-10: {x}' if str(x) != 'nan' else "",
        'PMSI_HOSP_Libelle_Diag English': lambda x: f"{x}",

    }

    col_to_str_fr = {
        # Hospit_mere
        'DELAI_HOSP': lambda x: f"- Entrée à l'hôpital au {x}ᵉ jour de la grossesse",
        'DAYS_HOSP': lambda x: f"Séjour à l'hôpital pendant {x} jours.",
        'PMSI_TYPE_DIAG': lambda x: '- Diagnostic principal' if int(x) == 1 else '- Diagnostic associé' if int(
            x) == 2 else '- Diagnostic relié',
        'PMSI_HOSP_DIAG': lambda x: f"CIM-10 {x}" if str(x) != 'nan' else "",
        'PMSI_HOSP_Libelle_Diag': lambda x: f"{x}",

    }
    build_txt_fr = [ 'PMSI_TYPE_DIAG', 'PMSI_HOSP_Libelle_Diag', 'PMSI_HOSP_DIAG']
    build_txt_en = [ 'PMSI_TYPE_DIAG', 'PMSI_HOSP_Libelle_Diag English', 'PMSI_HOSP_DIAG']

    to_remove_diagnostics = [
        'O35.0', #'Maternal care for (suspected) fetal central nervous system malformation',
        'O35.9', # 'Maternal care for (suspected) fetal anomaly and injury, unspecified',
        'O35.8', # 'Maternal care for other (suspected) fetal abnormalities and injuries',
        'O32.9', # 'Maternal care for abnormal fetal presentation, unspecified',
    ]

    for lang_template, lang_str, build_txt in zip([col_to_str, col_to_str_fr], ['en', 'fr'], [build_txt_en, build_txt_fr]):
        print("Language", lang_str)
        df_hosp = read_csv_hospitalization(db_path, lang=lang_str)
        df_hosp = df_hosp[~df_hosp['PMSI_HOSP_DIAG'].isin(to_remove_diagnostics)]
        df_main = read_csv_main(db_path)[['NUM_GROSSESSE', 'DATE_LMP']]
        df_hosp = df_hosp.merge(df_main, on='NUM_GROSSESSE', how='left')

        # filtered_df = df_hosp[df_hosp['PMSI_HOSP_Libelle_Diag'].str.contains('malfo', case=False, na=False)]
        # Add DATE_LMP from ISSUE
        nb_days_sejour = df_hosp['PMSI_HOSP_Dtsortie'] - df_hosp['PMSI_HOSP_DtEntree']
        df_hosp['DAYS_HOSP'] = [x.days for x in nb_days_sejour]
        delai_sejour = df_hosp['PMSI_HOSP_DtEntree'] - df_hosp['DATE_LMP']
        df_hosp['DELAI_HOSP'] = [x.days for x in delai_sejour]

        df_hosp.drop(columns=['PMSI_HOSP_Dtsortie', 'PMSI_HOSP_DtEntree', 'DATE_LMP'], inplace=True)
        for trim in [-1, 1, 2, 3]:
            if trim == -1:
                tmp_df = df_hosp[(df_hosp['DELAI_HOSP'] >= -31) & (df_hosp['DELAI_HOSP'] <= 97)]
            elif trim == 1:
                tmp_df = df_hosp[df_hosp['DELAI_HOSP'] <= 97]
            elif trim == 2:
                tmp_df = df_hosp[df_hosp['DELAI_HOSP'] <= 195]
            elif trim == 3:
                tmp_df = df_hosp

            text_df = {}
            text_df['NUM_GROSSESSE'] = tmp_df['NUM_GROSSESSE']
            text_df['NUM_MERE'] = tmp_df['NUM_MERE'].values
            text_df['DELAI_HOSP'] = tmp_df['DELAI_HOSP'].values
            text_df['DAYS_HOSP'] = tmp_df['DAYS_HOSP'].values
            text_df['txt'] = tmp_df.apply(lambda x: ". ".join([(lang_template[col])(x[col]) for col in build_txt
                                                               if len((lang_template[col])(x[col])) > 0])+".",axis=1)
            new_df = pd.DataFrame(text_df)
            #new_df.reset_index(drop=True, inplace=True)
            #new_df.set_index('NUM_GROSSESSE', inplace=True)

            def format_text(group):
                if lang_str == 'en':
                    delai = f'* Hospital entry at {group.name[1]}th pregnancy day'  # DELAI_HOSP value
                    days = f'Hospital stay during {group.name[2]} days'  # DAYS_HOSP value
                else:
                    delai = f"* Entrée à l'hôpital au {group.name[1]}ᵉ jour de la grossesse"
                    days = f"Séjour à l'hôpital pendant {group.name[2]} jours.",
                #base_text = f"Hospitalized after {delai} days for {days} days: "
                return f"{delai}. {days}" + ". ".join(group['txt'])


            # First sort by date, then group and concatenate by num_grossesse
            final_df = (new_df.sort_values('DELAI_HOSP').groupby(['NUM_GROSSESSE', 'DELAI_HOSP', 'DAYS_HOSP']).apply(format_text).reset_index(name='txt'))
            final_df = final_df.groupby('NUM_GROSSESSE').txt.agg('\n'.join)#.reset_index()
            print("Final dataset shape", final_df.shape)

            filename = f'cleaned_txt_hosp_t{trim}_{lang_str}.csv'
            if trim == -1:
                filename = f'cleaned_txt_hosp_t_window_{lang_str}.csv'
            print("Save generated data into ", filename)
            final_df.to_csv(os.path.join(output_path, filename), sep=';')#, quoting = csv.QUOTE_ALL, escapechar = "\\")



def analyze_data(db_path):
    df_issue = read_csv_main(db_path)
    # Remove multiple pregnancy
    df_issue = df_issue[df_issue['GROSSESSE_MULTIPLE'] == 0].drop(columns=['GROSSESSE_MULTIPLE'])
    df_prescription = read_csv_prescription(db_path)
    df_prescription = df_prescription.drop(columns=['DATE_LMP', 'DATE_DEL', 'ORGANO'])

    def analyze_prescriptions(my_prescription):
        # Counters for conditions
        last_word_counter = Counter()
        second_last_counter = Counter()
        all_last_word_counter = Counter()
        all_second_last_counter = Counter()

        # Extended analysis
        last_three_counter = Counter()
        count_not_number = 0

        for item in tqdm(my_prescription):
            words = item.split()
            last_word = words[-1]
            second_last = words[-2]
            all_last_word_counter[last_word] += 1
            all_second_last_counter[second_last] += 1
            if not last_word.isdigit():
                #print(item)
                count_not_number+=1
                continue
            # Count occurrences of last words
            last_word_counter[last_word] += 1
            second_last_counter[second_last] += 1

            # # If the last word is not a number, take the last three words
            # if not last_word.isdigit():
            #     last_three_counter[' '.join(words[-3:])] += 1


        #print("Occurrences of last words:", dict(last_word_counter))
        print("Occurrences of second last word:", sorted(dict(second_last_counter).items(), key=lambda kv: -kv[1]))
        print("Occurrences of ALL last words:", sorted(dict(all_last_word_counter).items(), key=lambda kv: -kv[1]))
        #print("Occurrences of ALL second last word:", dict(all_second_last_counter))
        print("Total prescriptions", len(my_prescription))
        print("Not finishing in a number", count_not_number)

    print("Analyze medicaments names")
    all_meds_names = df_prescription['NOM'].unique()
    analyze_prescriptions(all_meds_names)

    def get_stats_from_prescriptions(my_issue, my_prescription):
        full_prescription = my_issue.merge(my_prescription, on='NUM_GROSSESSE', how='left')
        total_atc = len(full_prescription['atc'].unique())
        total_cip = len(full_prescription['CIP'].unique())

        print("There are in total", len(full_prescription), " prescriptions in the full dataset")
        print(f"There in total {total_atc} atc codes and {total_cip} CIP codes")
        top_meds = full_prescription[['NOM']].value_counts()[:5]
        meds_per_patient = full_prescription.groupby('NUM_GROSSESSE')['NUM_GROSSESSE'].count()
        print(meds_per_patient.max(), meds_per_patient.min(), meds_per_patient.mean(), top_meds)

    print("Total prescriptions")
    get_stats_from_prescriptions(df_issue, df_prescription)

    print("Only Trimester 1")
    get_stats_from_prescriptions(df_issue, df_prescription[df_prescription['TRIM_PRESCRIPTION_SA']<2])

    print("Total prescriptions with only malformation cases")
    get_stats_from_prescriptions(df_issue[df_issue['MALFO_MAJEURE']==1], df_prescription)

    print("Only Trimester 1 with only malformation cases")
    get_stats_from_prescriptions(df_issue[df_issue['MALFO_MAJEURE']==1], df_prescription[df_prescription['TRIM_PRESCRIPTION_SA'] < 2])

    breakpoint()


if __name__ == '__main__':
    # analyze_data(root_db)

    # Atemporal information, issue data and geographic data
    prepare_profile_information(root_db)
    # Temporal information
    prepare_prescription_information_txt(root_db)
    prepare_hosp_information_txt(root_db)
