from Experiments import Experiment
import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiments')
    parser.add_argument('exp', help="Experiment name")
    args = parser.parse_args()

    if args.exp == "family_income":
    	# Read in the data
        data = pd.read_csv('../family_income/family_income.csv')
        columns = pd.read_csv('../family_income/column_types.csv')
        # Bucket age
        age_buckets = pd.cut(data["Household Head Age"],5).astype(str)
        data["Household Head Age"] = age_buckets
        groups = pd.unique(data["Household Head Age"])
        gender_groups = pd.unique(data["Household Head Sex"])
        # Change categories to numbers.
        features = []
        for index, row in columns.iterrows():
            if row.Name == 'Total Household Income':
                income = data[row.Name].values
            elif row.Name != "Household Head Age" and row.Name != "Household Head Sex":
                if row.Type == "vals":
                    data[row.Name] = data[row.Name].astype('category')
                    data[row.Name] = data[row.Name].cat.codes
                features.append(row.Name)
        print(features)
        # Group by groups and gender
        gk = data.groupby(['Household Head Age', 'Household Head Sex'])
        context_matrix = []
        reward_matrix = []
        group_names = []
        idx = 0
        groups = {'sensitive': [], 'not_sensitive': []}
        sensitive_group = {}
        for name, group in gk:
            group_names.append(name)
            if name[1] == "Female":
                groups['sensitive'].append(idx)
                sensitive_group[idx] = True
            else:
                groups['not_sensitive'].append(idx)
                sensitive_group[idx] = False
            context_matrix.append(group[features].values)
            reward_matrix.append(group['Total Household Income'].values)
            idx += 1
        arms = len(group_names)
        context_size = len(features)

        deltas = [.1, .2, .3, .4, .5]
        runs = 100
        Ts = [2 * n * arms for n in range(1, 21)]

        algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairTopInterval"]
        filename = "../experiments/%s/_context_%s" % (args.exp, context_size)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        experiment = Experiment(arms,context_size,groups,
						        bandit_types=algorithms,deltas=deltas,
						        Ts=Ts,arm_type="real",
						        filename=filename,sensitive_group=sensitive_group,
						        context_matrix=context_matrix, reward_matrix=reward_matrix)
        experiment.run_x_experiments(runs)
    elif args.exp == 'uci_student':
        # Read in the data
        data = pd.read_csv('../uci_student/student-por.csv', sep=';')
        features = ['school', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                   'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                   'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                   'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                   'Walc', 'health', 'absences']
        # Change categories to numbers
        categorical_columns = ['school', 'address', 'famsize', 'Pstatus', 'Mjob',
                       'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 
                       'paid', 'activities', 'nursery', 'higher', 'internet', 
                       'romantic']
        for cat in categorical_columns:
            data[cat], _ = data[cat].factorize()
        # Bucket age
        data.age[data.age>18] = 19 # 19 is now >18 basically - just a hacky way to do this
        groups = pd.unique(data["age"])
        gender_groups = pd.unique(data["sex"])
        gk = data.groupby(['age', 'sex'])

        context_matrix = []
        reward_matrix = []
        group_names = []
        idx = 0
        groups = {'sensitive': [], 'not_sensitive': []}
        sensitive_group = {}
        for name, group in gk:
            group_names.append(name)
            if name[1] == "M":
                groups['sensitive'].append(idx)
                sensitive_group[idx] = True
            else:
                groups['not_sensitive'].append(idx)
                sensitive_group[idx] = False
            context_matrix.append(group[features].values)
            reward_matrix.append(group['G3'].values)
            idx += 1
        arms = len(group_names)
        context_size = len(features)

        deltas = [.1, .2, .3, .4, .5]
        # deltas = [.1]
        runs = 100
        Ts = [2 * n * arms for n in range(1, 21)]
        # Ts = [2 * n * arms for n in range(20, 21)]

        algorithms = ["TopInterval", "IntervalChaining", "Random", "GroupFairTopInterval"]
        # algorithms = ["TopInterval"]
        filename = "../experiments/%s/_context_%s" % (args.exp, context_size)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        experiment = Experiment(arms,context_size,groups,
                                bandit_types=algorithms,deltas=deltas,
                                Ts=Ts,arm_type="real",
                                filename=filename,sensitive_group=sensitive_group,
                                context_matrix=context_matrix, reward_matrix=reward_matrix)
        experiment.run_x_experiments(runs)
