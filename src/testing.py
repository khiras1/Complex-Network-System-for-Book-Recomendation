import random
import time

import pandas as pd

from src.collaborative_filtering import get_recommendations_cf
from src.personalized_pagerank import get_recommendations_ppr
from src.utils import get_books_for_user


def get_recommendations_func(method):
    if method == "pagerank":
        return get_recommendations_ppr
    else:
        return lambda G, target_user, top_n: get_recommendations_cf(
            G=G, target_user=target_user, top_n=top_n, method=method
        )


def get_metrics_for_user(graph, test_user, get_recommendations, k, test_size):
    test_books = get_books_for_user(graph, test_user)
    random_sample = random.sample(
        sorted(test_books), k=int(len(test_books) * test_size)
    )
    for book in random_sample:
        graph.remove_edge(test_user, book)
    start = time.time()
    recommendations = get_recommendations(graph, test_user, top_n=k)
    execution_time = time.time() - start
    recommended_books = set([r[0] for r in recommendations])
    for book in recommended_books:
        graph.add_edge(test_user, book)
    return {
        "user": test_user,
        "precision": len(recommended_books.intersection(set(test_books))) / k,
        "recall": len(recommended_books.intersection(set(test_books)))
        / len(random_sample),
        "execution_time": execution_time,
    }


def get_metrics_at_k(graph, test_users, get_recommendations, k=15, test_size=0.2):
    log = []
    for user in test_users:
        log.append(
            get_metrics_for_user(
                graph, user, get_recommendations, k=k, test_size=test_size
            )
        )
    return pd.DataFrame(log)
