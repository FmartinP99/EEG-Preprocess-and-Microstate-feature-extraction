from itertools import combinations
import pandas as pd
import numpy as np
import csv
from pathlib import Path
import sys
import networkx as nx


def make_graph(df_path, date_str, limit=0.1, control_class=1, compare_class=2, below_limit=True, columns_to_drop=None, outdir="characteristics/graphs"):
    """
    Calculates the correlation matrix (lower triangular part) below <limits> and the graph for the make_cliques function.
    :param df_path: .csv to calculate the filtered corr_matrix from
    :param date_str: String to differentiate the result file. It is used for file naming purposes only.
    :param limit: Limit for the values in the corr matrix.
    :param cc: Control class.
    :param cc: Which class' row to keep besides the one labeled control class.
    :param below_limit: Collect the edges either below or above the limit.
    :param columns_to_drop: these columns will be excluded from the correlation map.
    :param outdir: Directory to write the result files.
    :return: None
    """

    if columns_to_drop is None:
        columns_to_drop = []


    df = pd.read_csv(df_path)
    if "label" in df.columns.tolist():
        df.rename(columns={"label": "class"}, inplace=True)

    if "category" in df.columns.tolist():
        df.rename(columns={"category": "class"}, inplace=True)

    df = df[(df["class"] == control_class) | (df["class"] == compare_class)]
    columns = df.columns

    for cla in columns_to_drop:
        if cla in columns:
            df.drop([cla], inplace=True, axis=1)

    print(df)
    columns = df.columns
    corr_matrix = abs(df[columns].corr())

    Path(f"{outdir}/corr_matrices").mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(f"{outdir}/corr_matrices/corr_matrix_{date_str}.csv")
    columns = df.columns.tolist()
    corr_values = corr_matrix.values.tolist()
    indexes_below_limit = []

    for i in range(1, len(corr_values)):
        for j in range(0, i):
            if below_limit is True:
                if corr_values[i][j] < limit:
                    dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j], "limit": str(round(limit, 2))}
                    indexes_below_limit.append(dict_to_append)
            elif below_limit is False:
                if corr_values[i][j] > limit:
                    dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j],
                                      "limit": str(round(limit, 2))}
                    indexes_below_limit.append(dict_to_append)
            else:
                print("'below_limit' argument should be either True or False!!")
                sys.exit()

    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/graph_{date_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["col1", "col2", "value", "limit"])
        writer.writeheader()
        writer.writerows(indexes_below_limit)



def make_graph2(df_path, date_str, limit=0.1, below_limit=True, columns_to_drop=None, outdir=r"characteristics\graphs"):
    """
    Calculates correlation matrix (lower triangular part) below <limits> for the make_cliques function. Use this, if the .csv at @df_path is a corr matrix itself.
    :param df_path: .csv to calculate the filtered corr_matrix from
    :param date_str: String to differentiate the result file. It is used for file naming purposes only.
    :param limit: Limit for the values in the corr matrix.
    :param below_limit: Collect the edges either below or above the limit.
    :param columns_to_drop: these columns will be excluded from the correlation map.
    :param outdir: Directory to write the result files.
    :return: None
    """

    if columns_to_drop is None:
        columns_to_drop = []

    df = pd.read_csv(df_path, index_col=0)
    columns = df.columns
    for cla in columns_to_drop:
        if cla in columns:
            df.drop([cla], inplace=True, axis=1)
            df.drop([cla], inplace=True, axis=0)


    corr_matrix = df

    corr_matrix.to_csv(f"{outdir}/korr_matrix{date_str}.csv")
    columns = df.columns.tolist()
    corr_values = corr_matrix.values.tolist()
    indexes_below_limit = []


    for i in range(1, len(corr_values)):
        for j in range(0, i):
            if below_limit is True:
                if corr_values[i][j] < limit:
                    dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j],
                                      "limit": str(round(limit, 2))}
                    indexes_below_limit.append(dict_to_append)
            elif below_limit is False:
                if corr_values[i][j] > limit:
                    dict_to_append = {"col1": columns[i], "col2": columns[j], "value": corr_values[i][j],
                                      "limit": str(round(limit, 2))}
                    indexes_below_limit.append(dict_to_append)
            else:
                print("'below_limit' argument should be either True or False!!")
                sys.exit()

    print(len(indexes_below_limit))
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/graph_{date_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["col1", "col2", "value", "limit"])
        writer.writeheader()
        writer.writerows(indexes_below_limit)



def get_disjunct_cliques(cliques):
    disjunkt_klikks = []

    for cg in sorted(cliques, key=len, reverse=True):
        if not disjunkt_klikks:
            disjunkt_klikks.append(set(cg))
        else:
            for dkl in disjunkt_klikks:
                if set(cg).issubset(dkl):
                    break
            else:
                disjunkt_klikks.append(set(cg))

    disjunkt_klikks = [list(d) for d in disjunkt_klikks]
    return disjunkt_klikks

def make_cliques(graph_df, date_str, min_clique=3, max_clique=9, disjunct_cliques=False, outdir="characteristics/cliques"):
    """
    This function searches for cliques in a graph, and writes them into a csv.
    :param graph_df: Dataframe containing the graph
    :param date_str: string to distingush the output file
    :param min_clique: minimum length cliques to search for
    :param max_clique: maximum length cliques to search for
    :param disjunct_cliques:  whether the program should not write sub-cliques of bigger cliques or not.
    :param outdir: path of the directory the program should write the output file
    :return: None
    """
    dfs = [graph_df[graph_df["limit"] == x] for x in graph_df.limit.unique().tolist()]

    headers = []
    list_to_write = []
    klikks_to_append = []

    for df in dfs:
        print(df)
        weights_dict = {}

        limit = df.limit.unique().tolist()[0]
        edges = pd.DataFrame({
            "source": df.col1.values,
            "target": df.col2.values,
            "weight": df.value.values,
        }
        )
        #####################

        vertices = set(edges.source.values)
        vertices.update(set(edges.target.values))
        vertices = list(vertices)
        found_vertices = [vertices[0]]
        sources = edges.source.values.tolist()
        targets = edges.target.values.tolist()

        index = 0

        while index < len(vertices):
            try:
                cs = found_vertices[index]
                for z1, z2 in zip(sources,targets):
                    if z1 == cs:
                        if z2 not in found_vertices:
                            found_vertices.append(z2)
                    elif z2 == cs:
                        if z1 not in found_vertices:
                            found_vertices.append(z1)
                index += 1
            except:
                break


        ######################
        if len(vertices) == len(found_vertices):

            G = nx.from_pandas_edgelist(edges, edge_attr=True)

            complete_graphs = [g for g in nx.enumerate_all_cliques(G) if len(g) > 2]
            print(f"Overall: {len(complete_graphs)}\n spreading:")
            klikks = []

            for _num_ in range(min_clique, max_clique):
                print(f"{_num_} -> {len([g for g in complete_graphs if len(g) == _num_])}")
                klikks.append(len([g for g in complete_graphs if len(g) == _num_]))

            klikks_to_append.append({"limit": limit,"data":klikks})
            print()

            disjunct_graphs = None
            if disjunct_cliques:
                disjunct_graphs = get_disjunct_cliques(complete_graphs)

                print("Disjunct cliques: ")
                for _num_ in range(min_clique, max_clique):
                    print(f"{_num_} -> {len([g for g in disjunct_graphs if len(g) == _num_])}")

            graphs_to_write = complete_graphs if not disjunct_cliques else disjunct_graphs

            for g in graphs_to_write:
                combs = combinations(g, 2)
                weights = []
                for c in combs:
                    if f"{c[0]}_{c[1]}" not in weights_dict.keys() and f"{c[1]}_{c[0]}" not in weights_dict.keys():
                        for index, row in edges.iterrows():
                            if (row["source"] == c[0] and row["target"] == c[1]) or (row["source"] == c[1] and row["target"] == c[0]):
                                if row["source"] == c[0] and row["target"] == c[1]:
                                    weights_dict[f"{c[0]}_{c[1]}"] = row["weight"]
                                elif row["source"] == c[1] and row["target"] == c[0]:
                                    weights_dict[f"{c[1]}_{c[0]}"] = row["weight"]
                                weights.append(row["weight"])
                    else:
                        if f"{c[0]}_{c[1]}" in weights_dict.keys():
                            weights.append(weights_dict[f"{c[0]}_{c[1]}"])
                        elif f"{c[1]}_{c[0]}" in weights_dict.keys():
                            weights.append(weights_dict[f"{c[1]}_{c[0]}"])
                max_weight = max(weights)
                min_weight = min(weights)
                avg_weight = np.mean(weights)
                std_weight = np.std(weights)

                dict_to_write = {}
                dict_to_write["limit"] = limit
                dict_to_write["feature"] = g
                dict_to_write["weights"] = weights
                dict_to_write["max_weight"] = max_weight
                dict_to_write["min_weight"] = min_weight
                dict_to_write["avg_weight"] = avg_weight
                dict_to_write["std_weight"] = std_weight
                dict_to_write["clique_length"] = len(g)
                list_to_write.append(dict_to_write)
        else:
            print(f"{limit} not general graph!!!")

    headers.append("feature")
    headers.append("weights")
    headers.append("max_weight")
    headers.append("min_weight")
    headers.append("avg_weight")
    headers.append("std_weight")
    headers.append("clique_length")
    headers.append("limit")
    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/clique_{date_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(list_to_write)



