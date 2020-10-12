import numpy as np
import pandas as pd

# Function for converting OTU ids to taxa info

def id_to_taxa(tax_df, zotu):
    order = ['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']
    tree_path = list(tax_df.loc[zotu])
    path_arr = np.array(tree_path)
    if type(path_arr[0]) == np.float64:
        print('{} has no taxanomic assignment'.format(zotu))
        return None
    nan_idx = np.where(path_arr == 'nan')[0]
    tree_path = np.delete(tree_path, nan_idx)
    order = np.delete(order, nan_idx)
    taxa = '|'.join([i+j for i,j in zip(order, tree_path)])
    return taxa


# Function for grouping the results dataframe by their parameters

def group_by_params(df, num_combinations, num_scores = 2):
    mean_score_df = pd.DataFrame(index = df.index[:num_scores], columns = [comb + 1 for comb in range(num_combinations)])
    params = []
    for i in range(num_combinations):
        filtered = df.filter(regex=("CV_.*_GS_{}$".format(i+1)))
        params.append(filtered.loc['Params', "CV_1_GS_{}".format(i+1)])
        mean_score_df[i+1] = filtered.loc[filtered.index[:num_scores]].mean(axis = 1).values
    mean_score_df.loc['Params'] = params
    return mean_score_df, params 