import numpy as np
import pandas as pd
import csv
import os
import sys
import copy
from pathlib import Path
import ast

def write_to_csv(list_to_write, headers, outdir, filename):
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/{filename}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)


def __calculate_sequences(l_seq):
    """calculates each clusters length,  {"A": [10,5,7,10,8],B:[...]} and returns it"""

    seq_dict = {}

    counter = 0
    _class = l_seq[0]
    for i in range(len(l_seq)):
        if _class == l_seq[i]:
            counter += 1
        else:
            if _class in seq_dict.keys():
                _l = seq_dict[_class]
                _l.append(counter)
                seq_dict[_class] = _l
            else:
                _l = [counter]
                seq_dict[_class] = _l
            counter = 1
            _class = l_seq[i]

            if i == len(l_seq) - 1:
                _l = [counter]
                seq_dict[_class] = _l

    return seq_dict


def calculate_statisticals_from_sequences(l_seq, name="", category=0):
    """ calculates the avg, median, max length of sequences
    :returns a dict with the datas
    """
    seq_dict = __calculate_sequences(l_seq)
    statistics_dict = {}
    for key, value in seq_dict.items():
        _mean = np.mean(value)
        _median = np.median(value)
        _max = np.max(value)
        _dict_to_append = {"seq_mean": _mean, "seq_median": _median, "seq_max": _max, "category": category, "id": name}
        statistics_dict[key] = _dict_to_append

    return statistics_dict


def calculate_transition_sequences(l_seq):
    """ calculates transitions "valueLength,valueLength" - pairs,  A4,B4,A1...  class 1 for 4 length, class 2 for 2 length, class 1 for 1 length...
    :returns string"""

    key_dict = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    #print(l_seq)

    counter = 0
    _class = l_seq[0]
    transition_str = ""
    for i in range(len(l_seq)):
        if _class == l_seq[i]:
            counter += 1
        else:
            if len(transition_str) > 0:
                transition_str = transition_str + f"{key_dict[_class]}{counter},"
            else:
                transition_str = f"{key_dict[_class]}{counter},"
            counter = 1
            _class = l_seq[i]
            if i == len(l_seq) - 1:
                if len(transition_str) > 0:
                    transition_str = transition_str + f"{key_dict[_class]}{counter},"
                else:
                    transition_str = f"{key_dict[_class]}{counter},"
    return transition_str


dirpath = "../preprocess/csvs/"

def read_file_characteristics(dirpath, list_of_categories, outdir="characteristics"):

    list_to_write_statistics = []
    list_to_write_transitions = []

    for ix, file in enumerate(os.listdir(dirpath)):
        print(f"{file} is in the making!")

        df = pd.read_csv(f"{dirpath}/{file}")
        name = file.split("_")[2]

        category = list_of_categories[ix]

        l_seq = df["microstates"].values.tolist()

        ######
        seq_stat = calculate_statisticals_from_sequences(l_seq, name, category)

        for key, value in seq_stat.items():
            micr = "A" if key == 1 else "B" if key == 2 else "C" if key == 3 else "D" if key == 4 else "E" if key == 5 else None
            value["microstate"] = micr
            list_to_write_statistics.append(value)
        ######

        ######
        seq_trans = calculate_transition_sequences(l_seq)
        dict_to_add_trans = {"id": name, "transitions": seq_trans, "category": category}
        list_to_write_transitions.append(dict_to_add_trans)
        ######

    headers = ["id", "microstate", "seq_mean", "seq_median", "seq_max", "category", ]
    write_to_csv(list_to_write_statistics, headers, outdir,
                 f"eeg_sequence_statistical")
    write_to_csv(list_to_write_transitions, ["id", "transitions", "category"], outdir,
                 f"eeg_sequence_transitions")

