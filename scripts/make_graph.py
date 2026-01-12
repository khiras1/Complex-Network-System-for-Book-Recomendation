import pickle
import pandas as pd
import networkx as nx

THRESHOLD = 20


def make_bipartite_graph(data):
    """
    Create a bipartite graph from the interactions DataFrame.
    Users are assigned bipartite=0 and books bipartite=1.
    """
    B = nx.Graph()
    # Add user nodes (partition 0)
    users = data["user_id"].unique()
    B.add_nodes_from(users, bipartite=0)

    # Add book nodes (partition 1)
    books = data["book_id"].unique()
    B.add_nodes_from(books, bipartite=1)

    # Add edges between users and books
    for _, row in data.iterrows():
        B.add_edge(row["user_id"], row["book_id"])

    return B


def main():
    # Load data
    data = pd.read_csv("data/goodreads_interactions.csv")
    print("Data loaded")

    # Keep only reviewed and read interactions
    data = data[(data["is_reviewed"] == 1) & (data["is_read"] == 1)]

    # Remove rows where rating is 1 or 2 or 3
    data = data[~data["rating"].isin([1, 2, 3])]

    # Filter out books with fewer than THRESHOLD reviews
    book_counts = data["book_id"].value_counts()
    books_to_keep = book_counts[book_counts > THRESHOLD].index
    data = data[data["book_id"].isin(books_to_keep)]

    # Filter out users with fewer than THRESHOLD reviews
    user_counts = data["user_id"].value_counts()
    users_to_keep = user_counts[user_counts > THRESHOLD].index
    data = data[data["user_id"].isin(users_to_keep)]

    print("Data filtered")

    # change user id to u_id_ and book id to b_id_
    data["user_id"] = "u_id_" + data["user_id"].astype(str)

    # Create a bipartite graph
    B = make_bipartite_graph(data)
    print("Bipartite graph created")

    print(nx.is_bipartite(B))

    # Extract largest connected component
    largest_cc = max(nx.connected_components(B), key=len)
    subgraph = B.subgraph(largest_cc).copy()

    # Save the subgraph
    with open("data/bipartite_graph.gpickle", "wb") as f:
        pickle.dump(subgraph, f)
    print("Largest connected bipartite component saved to data/bipartite_graph.gpickle")

    B_df = nx.to_pandas_edgelist(B, source="users", target="books")
    B_df.to_pickle("data/bipartite_graph_df.pkl")
    print("Edgelist dataframe saved to data/bipartite_graph_df.pkl")


if __name__ == "__main__":
    main()
