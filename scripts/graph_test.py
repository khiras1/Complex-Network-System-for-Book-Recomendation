import json
import random

import pandas as pd

from src.testing import get_recommendations_func, get_metrics_at_k
from src.utils import create_bipartite_from_pkl


def main(n_users_to_test):
    methods = ["jaccard", "overlap", "adamic_adar", "resource_allocation", "pagerank"]

    path_to_graph = "data/bipartite_graph.gpickle"
    graph = create_bipartite_from_pkl(path_to_graph)
    print("loaded graph")

    all_users = [n for n, d in graph.nodes(data=True) if d.get("bipartite") == "users"]
    test_users = random.sample(all_users, k=n_users_to_test)

    output = {}
    for method in methods:
        get_recommendations = get_recommendations_func(method)
        log = pd.DataFrame(
            get_metrics_at_k(
                graph, test_users, get_recommendations=get_recommendations, k=15
            )
        )
        log.to_csv("output/methods/{}.csv".format(method))
        output[method] = {
            "precision": log["precision"].mean(),
            "recall": log["recall"].mean(),
            "execution_time": log["execution_time"].mean(),
        }
        print("tested method: {}".format(method))

    with open("output/precision_for_method.json", "w") as fp:
        json.dump(output, fp, indent=4)
    print("saved output")


if __name__ == "__main__":
    random.seed(42)
    N = 150
    main(N)
