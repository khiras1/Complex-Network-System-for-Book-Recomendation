import random
import pickle
from typing import Optional

import networkx as nx


def create_random_bipartite_graph(
    n_users: int = 10,
    n_books: int = 15,
    p_edge: float = 0.2,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Create a random bipartite graph with users and books.

    Args:
        n_users (int, optional): Number of users. Defaults to 10.
        n_books (int, optional): Number of books. Defaults to 15.
        p_edge (float, optional): Probability of edge creation. Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        nx.Graph: A bipartite graph with users and books.
    """
    if seed is not None:
        random.seed(seed)
    G = nx.Graph()
    for i in range(n_users):
        G.add_node(f"u{i}", bipartite="users")
    for j in range(n_books):
        G.add_node(f"k{j}", bipartite="books")
    for i in range(n_users):
        for j in range(n_books):
            if random.random() < p_edge:
                G.add_edge(f"u{i}", f"k{j}")
    return G


def create_bipartite_from_pkl(pkl_path: str, users_key: int = 0, books_key: int = 1):
    """Create a bipartite graph from a pickle file.

    Args:
        pkl_path (str): Path to the pickle file containing the graph.
        users_key (int, optional): Key for user nodes in the pickled graph. Defaults to 0.
        books_key (int, optional): Key for book nodes in the pickled graph. Defaults to 1.

    Returns:
        _type_: _description_
    """

    with open(pkl_path, "rb") as f:
        G = pickle.load(f)

    user_nodes = [n for n, data in G.nodes(data=True) if data["bipartite"] == users_key]
    book_nodes = [n for n, data in G.nodes(data=True) if data["bipartite"] == books_key]
    B = nx.Graph()
    B.add_nodes_from(user_nodes, bipartite="users")
    B.add_nodes_from(book_nodes, bipartite="books")
    B.add_edges_from(G.edges())

    return B


def get_users_for_book(G: nx.Graph, book: str) -> set:
    """Get users who have read a specific book.

    Args:
        G (nx.Graph): The bipartite graph.
        book (str): The book node.

    Returns:
        set: A set of user nodes who have read the book.
    """
    return {n for n in G.neighbors(book) if G.nodes[n]["bipartite"] == "users"}


def get_books_for_user(G: nx.Graph, user: str) -> set:
    """Get books read by a specific user.

    Args:
        G (nx.Graph): The bipartite graph.
        user (str): The user node.

    Returns:
        set: A set of book nodes read by the user.
    """
    return {n for n in G.neighbors(user) if G.nodes[n]["bipartite"] == "books"}
