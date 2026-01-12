import math
import heapq
from typing import List, Tuple, Optional

import networkx as nx

from src.utils import get_users_for_book, get_books_for_user


def jaccard_similarity(users_b1: set, users_b2: set) -> float:
    """Compute Jaccard similarity between two books based on common users.

    Args:
        users_b1 (set): Users who read the first book.
        users_b2 (set): Users who read the second book.

    Returns:
        float: Jaccard similarity score.
    """
    if not users_b1 and not users_b2:
        return 0.0
    intersection = users_b1 & users_b2
    union = users_b1 | users_b2
    return len(intersection) / len(union)


def overlap_coefficient(users_b1: set, users_b2: set) -> float:
    """
    Compute overlap coefficient between two books based on common users.

    Args:
        users_b1 (set): Users who read the first book.
        users_b2 (set): Users who read the second book.

    Returns:
        float: Overlap coefficient score.
    """
    if not users_b1 or not users_b2:
        return 0.0
    intersection = users_b1 & users_b2
    min_size = min(len(users_b1), len(users_b2))
    return len(intersection) / min_size


def adamic_adar_similarity(users_b1: set, users_b2: set, books_cache: dict) -> float:
    """
    Compute Adamic-Adar similarity between two books based on common users.

    Args:
        users_b1 (set): Users who read the first book.
        users_b2 (set): Users who read the second book.
        books_cache (dict): Cache of books read by users.

    Returns:
        float: Adamic-Adar similarity score.
    """
    common_users = users_b1 & users_b2
    if not common_users:
        return 0.0
    aa_sum = 0.0
    for user in common_users:
        degree = len(books_cache[user])
        if degree > 1:
            aa_sum += 1.0 / math.log(degree)
    return aa_sum


def resource_allocation_similarity(
    users_b1: set, users_b2: set, books_cache: dict
) -> float:
    """
    Compute Resource Allocation similarity between two books based on common users.

    Args:
        users_b1 (set): Users who read the first book.
        users_b2 (set): Users who read the second book.
        books_cache (dict): Cache of books read by users.

    Returns:
        float: Resource Allocation similarity score.
    """
    common_users = users_b1 & users_b2
    if not common_users:
        return 0.0
    ra_sum = 0.0
    for user in common_users:
        degree = len(books_cache[user])
        if degree > 0:
            ra_sum += 1.0 / degree
    return ra_sum


def get_recommendations_cf(
    G: nx.Graph, target_user: str, method: str = "jaccard", top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Get book recommendations for a user based on collaborative filtering.

    Args:
        G (nx.Graph): The bipartite graph.
        target_user (str): The user node for whom to get recommendations.
        method (str, optional): Similarity method to use. Defaults to "jaccard".
        top_n (int, optional): Number of recommendations to return. Defaults to 5.

    Raises:
        ValueError: If the similarity method is unknown.

    Returns:
        List[Tuple[str, float]]: A list of recommended books and their scores.
    """
    read_books = {
        n for n in G.neighbors(target_user) if G.nodes[n]["bipartite"] == "books"
    }

    # Precompute users for each book and books for each user
    users_cache = {}
    books_cache = {}
    for node, data in G.nodes(data=True):
        if data["bipartite"] == "books":
            users_cache[node] = get_users_for_book(G, node)
        elif data["bipartite"] == "users":
            books_cache[node] = get_books_for_user(G, node)

    if method == "jaccard":
        similarity_func = lambda b1, b2: jaccard_similarity(
            users_cache[b1], users_cache[b2]
        )
    elif method == "overlap":
        similarity_func = lambda b1, b2: overlap_coefficient(
            users_cache[b1], users_cache[b2]
        )
    elif method == "adamic_adar":
        similarity_func = lambda b1, b2: adamic_adar_similarity(
            users_cache[b1], users_cache[b2], books_cache
        )
    elif method == "resource_allocation":
        similarity_func = lambda b1, b2: resource_allocation_similarity(
            users_cache[b1], users_cache[b2], books_cache
        )
    else:
        raise ValueError(
            "Unknown similarity method. Allowed: 'jaccard', 'overlap', 'adamic_adar', 'resource_allocation'"
        )

    candidate_scores = {}

    # Collect all candidate books via co-readers (excluding already read)
    candidate_books = set()
    for b in read_books:
        users = users_cache[b] - {target_user}
        for u in users:
            candidate_books.update(books_cache[u])
    candidate_books -= read_books

    # Compute similarity scores only for candidate books
    for candidate in candidate_books:
        score = 0.0
        for b in read_books:
            sim = similarity_func(b, candidate)
            score += sim
        if score > 0:
            candidate_scores[candidate] = score

    # Efficient top-N selection
    recommendations = heapq.nlargest(
        top_n, candidate_scores.items(), key=lambda x: x[1]
    )
    return recommendations
