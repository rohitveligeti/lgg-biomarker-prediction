# Rohit Veligeti
# December 12th 2021

# This python file generates ML-ready datasets using cBioPortal's data on TCGA's & the PanCancer Atlas' study of Low-Grade Glioma
# The exception is the followup and patient data which was retrieved from the GDC Data Commons

from numpy.core.fromnumeric import argpartition
import pandas

# This function gets specific patient-related data

def get_patient_data():
    
    raw_patient_data = pandas.read_csv('clinical.txt', delimiter='\t', index_col=1, header=1)
    raw_patient_data = raw_patient_data.drop(['CDE_ID:2003301'], axis=0)

    # We use the next command to filter for the patient information we care about because it is a dependent variable or we need to control for it
    raw_patient_data = raw_patient_data[['year_of_initial_pathologic_diagnosis', 'age_at_initial_pathologic_diagnosis', 'eastern_cancer_oncology_group', 
    'karnofsky_performance_score', 'preoperative_corticosteroids', 'headache_history']]

    # To make this data ready for ML, we need to do a little bit of preprocessing

    # Let's replace the unknown values in 'karnofsky_performance_score' with the mean of 87.47
    raw_patient_data['karnofsky_performance_score'] = raw_patient_data['karnofsky_performance_score'].replace("[Not Available]", 87.47)
    raw_patient_data['karnofsky_performance_score'] = raw_patient_data['karnofsky_performance_score'].replace("[Unknown]", 87.47)
    raw_patient_data['karnofsky_performance_score'] = raw_patient_data['karnofsky_performance_score'].replace("[Not Evaluated]", 87.47)

    # Let's replace the unknown values in 'eastern_cancer_oncology_group' with the mean of 0.57
    raw_patient_data['eastern_cancer_oncology_group'] = raw_patient_data['eastern_cancer_oncology_group'].replace("[Not Available]", 0.57)
    raw_patient_data['eastern_cancer_oncology_group'] = raw_patient_data['eastern_cancer_oncology_group'].replace("[Unknown]", 0.57)
    raw_patient_data['eastern_cancer_oncology_group'] = raw_patient_data['eastern_cancer_oncology_group'].replace("[Not Evaluated]", 0.57)

    # Let's replace all values other than 'YES' in 'preoperative_corticosteroids' and 'headache_history' with 0 and values of 'YES' with 1
    raw_patient_data['preoperative_corticosteroids'] = raw_patient_data['preoperative_corticosteroids'].replace("[Not Available]", 0)
    raw_patient_data['preoperative_corticosteroids'] = raw_patient_data['preoperative_corticosteroids'].replace("[Unknown]", 0)
    raw_patient_data['preoperative_corticosteroids'] = raw_patient_data['preoperative_corticosteroids'].replace("[Not Evaluated]", 0)
    raw_patient_data['preoperative_corticosteroids'] = raw_patient_data['preoperative_corticosteroids'].replace("NO", 0)
    raw_patient_data['preoperative_corticosteroids'] = raw_patient_data['preoperative_corticosteroids'].replace("YES", 1)

    raw_patient_data['headache_history'] = raw_patient_data['headache_history'].replace("[Not Available]", 0)
    raw_patient_data['headache_history'] = raw_patient_data['headache_history'].replace("[Unknown]", 0)
    raw_patient_data['headache_history'] = raw_patient_data['headache_history'].replace("[Not Evaluated]", 0)
    raw_patient_data['headache_history'] = raw_patient_data['headache_history'].replace("NO", 0)
    raw_patient_data['headache_history'] = raw_patient_data['headache_history'].replace("YES", 1)

    # There we have our patient data
    return raw_patient_data

# This function gets the supplemental hypoxia data

def get_hypoxia_data():

    # The data is formatted very nicely for us, so there won't be much work here
    raw_hypoxia_data = pandas.read_csv('data_clinical_supp_hypoxia.txt', delimiter='\t', index_col=0, header=0)

    return raw_hypoxia_data


# This function gets the armlevel CNA data 

def get_armlevel_cna_data(filter=None):

    raw_armlevel_cna_data = pandas.read_csv('data_armlevel_cna.txt', delimiter='\t', index_col=1, header=0)
    raw_armlevel_cna_data = raw_armlevel_cna_data.drop(['ENTITY_STABLE_ID', 'DESCRIPTION'], axis=1)

    # We want to replace gain with 1, loss with -1, and all others with 0
    raw_armlevel_cna_data = raw_armlevel_cna_data.replace("Unchanged", 0)
    raw_armlevel_cna_data = raw_armlevel_cna_data.fillna(0)
    raw_armlevel_cna_data = raw_armlevel_cna_data.replace("Gain", 1)
    raw_armlevel_cna_data = raw_armlevel_cna_data.replace("Loss", -1)

    # There are some inconsistencies with the data types, so we will convert them all to float
    raw_armlevel_cna_data = raw_armlevel_cna_data.astype(int)

    # For compatibility with the other datasets, we will transpose this dataset
    raw_armlevel_cna_data = raw_armlevel_cna_data.T

    index_values = raw_armlevel_cna_data.index.values.tolist()
    new_index_values = []

    for i in index_values:
        new_index_values.append(i[0:-3])

    raw_armlevel_cna_data.index = new_index_values

    if filter == None:
        return raw_armlevel_cna_data
    else:
        return raw_armlevel_cna_data[filter]


# This function gets the CNA data

def get_cna_data(filter=None, frequency_cutoff=10):

    raw_cna_data = pandas.read_csv('data_cna.txt', delimiter='\t', index_col=0, header=0)
    raw_cna_data = raw_cna_data.drop(['Entrez_Gene_Id'], axis=1)

    # For compatibility with the other datasets, we will transpose this dataset and fill any NA values preemptively

    raw_cna_data = raw_cna_data.T

    index_values = raw_cna_data.index.values.tolist()
    new_index_values = []

    for i in index_values:
        new_index_values.append(i[0:-3])

    raw_cna_data.index = new_index_values

    raw_cna_data = raw_cna_data.fillna(0)

    raw_cna_data = raw_cna_data.loc[:, (raw_cna_data == 0).sum() <= len(raw_cna_data.index) - frequency_cutoff]

    
    if filter == None:
        return raw_cna_data
    else:
        return raw_cna_data[filter]

# This function gets the mutation data

def get_mutation_data(filter=None, frequency_cutoff=10):

    raw_mutation_data = pandas.read_csv('data_mutations.txt', delimiter='\t', index_col=16, header=0)

    symbol = raw_mutation_data['SYMBOL'].value_counts().to_dict()
    use_symbols = []

    i = 0
    for key, value in symbol.items():
        i += 1

        use_symbols.append(key)

        if value < frequency_cutoff:
            break

    mutation_data = pandas.DataFrame(index=raw_mutation_data.index.value_counts().to_dict().keys(), columns=use_symbols)

    for row in raw_mutation_data.iterrows():
        person = row[0]

        mutation_name = row[1]['Hugo_Symbol']

        if mutation_name in use_symbols:
            mutation_data.at[person, mutation_name] = 1

    mutation_data = mutation_data.fillna(0)
    mutation_data = mutation_data.astype(int)

    index_values = mutation_data.index.values.tolist()
    new_index_values = []

    for i in index_values:
        new_index_values.append(i[0:-3])

    mutation_data.index = new_index_values

    if filter == None:
        return mutation_data
    else:
        return mutation_data[filter]

# This function gets the patients' gliomas progression

def get_followup_data():

    raw_followup_data = pandas.read_csv('nationwidechildrens.org_clinical_follow_up_v1.0_lgg.txt', delimiter='\t', index_col=1, header=1)
    raw_followup_data = raw_followup_data.drop(['CDE_ID:2003301'], axis=0)

    raw_followup_data['primary_therapy_outcome_success'] = raw_followup_data['primary_therapy_outcome_success'].replace("[Not Available]", "Unknown")
    raw_followup_data['primary_therapy_outcome_success'] = raw_followup_data['primary_therapy_outcome_success'].replace("[Unknown]", "Unknown")
    raw_followup_data['primary_therapy_outcome_success'] = raw_followup_data['primary_therapy_outcome_success'].replace("[Discrepancy]", "Unknown")
    raw_followup_data['primary_therapy_outcome_success'] = raw_followup_data['primary_therapy_outcome_success'].replace("[Not Applicable]", "Unknown")

    return raw_followup_data['primary_therapy_outcome_success']

# This function aggregates all the previous functions' outcomes

def aggregrate(armlevel_filter=None, cna_filter=None, mutation_filter=None, cna_cutoff=50, mutation_cutoff=2):

    armlevel_cna_data = get_armlevel_cna_data(armlevel_filter)
    armlevel_cna_data = armlevel_cna_data.add_prefix("ALT_ARM_")

    cna_data = get_cna_data(filter=cna_filter, frequency_cutoff=cna_cutoff)
    cna_data = cna_data.add_prefix("ALT_CNA_")

    mutation_data = get_mutation_data(filter=mutation_filter, frequency_cutoff=mutation_cutoff)
    mutation_data = mutation_data.add_prefix("ALT_MUT_")
    follow_up_data = get_followup_data()

    armlevel_cna_set = set(armlevel_cna_data.index.values.tolist())
    cna_set = set(cna_data.index.values.tolist())
    mutation_set = set(mutation_data.index.values.tolist())
    follow_up_set = set(follow_up_data.index.values.tolist())

    mega_intersect = list(armlevel_cna_set.intersection(cna_set).intersection(mutation_set).intersection(follow_up_set))

    armlevel_cna_data = armlevel_cna_data.loc[mega_intersect]
    cna_data = cna_data.loc[mega_intersect]
    mutation_data = mutation_data.loc[mega_intersect]
    follow_up_data = follow_up_data.loc[mega_intersect]

    follow_up_data = follow_up_data.reset_index()
    follow_up_data = follow_up_data.drop_duplicates(subset='bcr_patient_barcode')
    follow_up_data = follow_up_data.set_index('bcr_patient_barcode')

    ndf = pandas.concat([armlevel_cna_data, cna_data, mutation_data], axis=1)



    ndf['Outcome'] = follow_up_data

    return ndf



