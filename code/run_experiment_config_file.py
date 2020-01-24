from Experiments import Experiment
import json
import argparse
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs Group Fair MAB experiments')
    parser.add_argument('config_file', help="Config File")
    args = parser.parse_args()

    # Get the config file
    with open(args.config_file, 'rb') as f:
        config = json.load(f)

    # Get the data
    data = pd.read_csv(config["data"]["filename"], sep=config["data"]["sep"])
    # Change categories to numbers
    for cat in config["categorical_features"]:
        data[cat], _ = data[cat].factorize()

    # Make the buckets
    if config["bucket"]["bucket_type"] == "ranges":
        data["temp_buckets"] = 0
        for rng in config["bucket"]["ranges"]:
            indeces = (data[config["bucket"]["column_name"]] >= rng[0]) & (data[config["bucket"]["column_name"]] < rng[1])
            data["temp_buckets"][indeces] = rng[0]
        data[config["bucket"]["column_name"]] = data["temp_buckets"]
    elif config["bucket"]["bucket_type"] == "auto":
        buckets = pd.cut(data[config["bucket"]["column_name"]],config["bucket"]["num_buckets"]).astype(str)
        data[config["bucket"]["column_name"]] = age_buckets

    # Group data
    groups = pd.unique(data[config["bucket"]["column_name"]])
    gender_groups = pd.unique(data[config["sensitive_group"]["column_name"]])
    gk = data.groupby([config["bucket"]["column_name"], config["sensitive_group"]["column_name"]])

    # Organize the data for the experiments
    context_matrix = []
    reward_matrix = []
    group_names = []
    idx = 0
    groups = {'sensitive': [], 'not_sensitive': []}
    sensitive_group = {}
    for name, group in gk:
        group_names.append(name)
        if name[1] == config["sensitive_group"]["sensitive_name"]:
            groups['sensitive'].append(idx)
            sensitive_group[idx] = True
        else:
            groups['not_sensitive'].append(idx)
            sensitive_group[idx] = False
        context_matrix.append(group[config["features"]].values)
        reward_matrix.append(group[config["reward_column"]].values)
        idx += 1
    arms = len(group_names)
    context_size = len(config["features"])

    if config["Ts"]["T_type"] == "range":
        Ts = [2 * n * arms for n in range(config["Ts"]["range"][0], config["Ts"]["range"][1])]
    else:
        Ts = config["Ts"]["list"]

    filename = "../experiments/%s/exp" % (config['experiment_name'])
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    experiment = Experiment(arms,context_size,groups,
                            bandit_types=config["algorithms"],deltas=config["deltas"],
                            Ts=Ts,arm_type="real",
                            filename=filename,sensitive_group=sensitive_group,
                            context_matrix=context_matrix, reward_matrix=reward_matrix)
    experiment.run_x_experiments(config["runs"])