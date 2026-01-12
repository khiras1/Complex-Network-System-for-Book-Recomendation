import heapq
from typing import List, Tuple

import networkx as nx


def get_recommendations_ppr(
    G: nx.Graph, target_user: str, top_n: int = 5, alpha: float = 0.85
) -> List[Tuple[str, float]]:
    """
    Get book recommendations for a user using personalized PageRank (optimized for large graphs).

    Args:
        G (nx.Graph): The bipartite graph.
        target_user (str): The user node for whom to get recommendations.
        top_n (int, optional): Number of recommendations to return. Defaults to 5.
        alpha (float, optional): Damping factor for PageRank. Defaults to 0.85.

    Returns:
        list: A list of recommended books with their PageRank scores.
    """
    personalization = {target_user: 1.0}
    pr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization)
    read_books = {
        n for n in G.neighbors(target_user) if G.nodes[n]["bipartite"] == "books"
    }

    candidate_books = (
        (node, score)
        for node, score in pr_scores.items()
        if G.nodes[node]["bipartite"] == "books" and node not in read_books
    )

    recommendations = heapq.nlargest(top_n, candidate_books, key=lambda x: x[1])
    return recommendations
