from Experiments import Experiment
import os
import argparse
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

sns.set(style="whitegrid")
# tips = sns.load_dataset("tips")
# print(tips)
# ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
#                   data=tips, palette="Set3")
# plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiments')
    parser.add_argument('exp', help="Experiment name")
    args = parser.parse_args()

    if args.exp == "family_income":
        # Read in the data
        data = pd.read_csv('../family_income/family_income.csv')
        columns = pd.read_csv('../family_income/column_types.csv')
        # Bucket age
        age_buckets = pd.cut(data["Household Head Age"],5)
        data["Household Head Age"] = age_buckets
        groups = pd.unique(data["Household Head Age"])
        gender_groups = pd.unique(data["Household Head Sex"])

        # ax = sns.boxplot(x="Household Head Age", y="Total Household Income", hue="Household Head Sex",
        #           data=data, palette="Set3", whis=5,showfliers=False)
        ax = sns.boxplot(x="Household Head Sex", y="Total Household Income",
                  data=data, palette="Set3", whis=5,showfliers=False)
        plt.show()
 #        # Change categories to numbers.
 #        features = []
 #        for index, row in columns.iterrows():
 #            if row.Name == 'Total Household Income':
 #                income = data[row.Name].values
 #            elif row.Name != "Household Head Age" and row.Name != "Household Head Sex":
 #                if row.Type == "vals":
 #                    data[row.Name] = data[row.Name].astype('category')
 #                    data[row.Name] = data[row.Name].cat.codes
 #                features.append(row.Name)
 #        print(features)
 #        # Group by groups and gender
 #        gk = data.groupby(['Household Head Age', 'Household Head Sex'])
 #        context_matrix = []
 #        reward_matrix = []
 #        group_names = []
 #        idx = 0
 #        groups = {'sensitive': [], 'not_sensitive': []}
 #        sensitive_group = {}
 #        things = {}
 #        for name, group in gk:
 #            group_names.append(name)
 #            mean = np.mean(group['Total Household Income'].values)
 #            std = np.std(group['Total Household Income'].values)
 #            mx = np.max(group['Total Household Income'].values)
 #            mn = np.min(group['Total Household Income'].values)
 #            if name[0] in things:
 #                things[name[0]][name[1]] = (mn,std)
 #            else:
 #                things[name[0]] = {name[1]: (mn,std)}
 #            if name[1] == "Female":
 #                groups['sensitive'].append(idx)
 #                sensitive_group[idx] = True
 #            else:
 #                groups['not_sensitive'].append(idx)
 #                sensitive_group[idx] = False
 #            context_matrix.append(group[features].values)
 #            reward_matrix.append(group['Total Household Income'].values)
 #            idx += 1
 #        print(things)
 # of the bars

       
