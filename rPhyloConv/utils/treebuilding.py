import numpy as np
import pandas as pd

from sklearn.preprocessing import scale, minmax_scale, MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from Bio import Phylo

class TreeBuilder(TransformerMixin, BaseEstimator):
    
    def __init__(self, tree_path):
        self.tree = Phylo.read(tree_path, format = "newick")
        self.clades = self.tabulate_names()
    
    def get_row(self, x):
        return x[1]
    
    def get_col(self, x):
        return x[0]

    def get_height(self):
        return len(self.level_dict.keys())

    def get_width(self):
        return max(list(map(self.get_length, self.level_dict.values())))

    def get_length(self, x):
        return len(x)
            
    def is_level_populated(self, x):
        return x[1] != []
    
    def is_list_empty(self, inList):
        if isinstance(inList, list):
            return all(map(self.is_list_empty, inList))
        return False
    
    def tabulate_names(self):
        names = {}
        for idx, clade in enumerate(self.tree.find_clades()):
            if clade.name:
                clade.name = "%d_%s" % (idx, clade.name)
            else:
                clade.name = str(idx)
            names[clade.name] = clade
        return names

    def get_level_dict(self):
        self.level_dict = {}
        level = 0
        at_bottom = False
        new_children = []
        child_name_list = []

        root = list(self.tree.find_clades(order="level"))[0]
        last_children = root.clades
        while not at_bottom:
            level+= 1
            for child in last_children:
                child_name = child.name
                if len(child_name.split("_")) == 2:
                    zotu_name = child_name.split("_")[1]
                    if zotu_name in self.features:
                        child_name_list.append(zotu_name)
                child_children = child.clades
                new_children+= child_children
            if self.is_list_empty(new_children):
                at_bottom = True
            self.level_dict[level] = child_name_list
            last_children = new_children
            new_children = []
            child_name_list = []
        self.level_dict = dict(filter(self.is_level_populated, self.level_dict.items()))
        return self
    
    def create_lookup_table(self):
        self.lookup_table = {}
        y = 0
        for key, value in self.level_dict.items():
            for zotu in value:
                x = value.index(zotu)
                self.lookup_table[zotu] = (x, y)
            y += 1
        return self
    
    def samples_to_matrices(self, samples):
        phylo_matrix_list = []
        rows = list(map(self.get_row, self.lookup_table.values()))
        cols = list(map(self.get_col, self.lookup_table.values()))
        for i in range(len(samples)):
            sample = samples.iloc[i]
            phylo_matrix = np.zeros((self.get_height(), self.get_width()))
            abundances = sample.loc[list(self.lookup_table.keys())].values
            phylo_matrix[rows, cols] = abundances
            phylo_matrix_list.append(phylo_matrix)
        phylo_matrices = np.array(phylo_matrix_list)
        return phylo_matrices
    
    def fit(self, X, y):
        self.features = list(X.columns)
        self.get_level_dict()
        self.create_lookup_table()
        X = self.samples_to_matrices(X)
        X = np.log(X+1)
        self.scaler = MinMaxScaler()
        original_shape = X.shape
        self.scaler.fit(X.reshape(-1, original_shape[1]*original_shape[2]))
        return self
        
    def transform(self, X):
        X = self.samples_to_matrices(X)
        X = np.log(X+1)
        original_shape = X.shape
        X = self.scaler.transform(X.reshape(-1, original_shape[1]*original_shape[2])).reshape(original_shape)
        X = np.expand_dims(X, axis = 3)
        X = np.clip(X, 0, 1)
        return X