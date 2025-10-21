
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
try: # try Louvain if available (optional)
    import community as community_louvain  # package: python-louvain
    HAVE_LOUVAIN = True
except Exception:
    HAVE_LOUVAIN = False


# =============================================
# Build a student-to-student graph from the activity
# matrix (students x resources) and analyze it
# =============================================

def build_similarity_matrix(X):
    """
    Compute cosine similarity between students
    -> X is students x resources
    -> return the dense similarity matrix 
    """
    sim = cosine_similarity(X.values.astype(float))
    np.fill_diagonal(sim, 0.0)  # for not link a node to itself
    return sim


def build_knn_graph(sim, student_ids, k= 10, sim_threshold= 0.2):
    """
    Build a k-NN graph:
    - for each student, connect to its top-k most similar peers
    - keep only edges with similarity >= sim_threshold
    """
    n = sim.shape[0]
    G = nx.Graph()
    G.add_nodes_from(student_ids)

    k = max(1, min(k, n - 1))
    for i in range(n):
        # find indices of top-k similarities for student i
        nbr_idx = np.argpartition(-sim[i], k)[:k]
        for j in nbr_idx:
            if i == j:
                continue
            w = float(sim[i, j])
            if w >= sim_threshold:
                u, v = student_ids[i], student_ids[j]
                # add undirected edge with weight
                if G.has_edge(u, v):
                    G[u][v]["weight"] = max(G[u][v]["weight"], w)
                else:
                    G.add_edge(u, v, weight=w)
    return G


def detect_communities(G):
    """
    detect communities in the graph.
    use Louvain if available, else NetworkX greedy modularity.
    """
    if HAVE_LOUVAIN:
        part = community_louvain.best_partition(G, weight="weight", random_state=42)
        return part
    # fallback (no extra install)
    comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    part = {}
    for cid, nodes in enumerate(comms):
        for u in nodes:
            part[u] = cid
    return part


def compute_modularity(G, partition):
    """
    Compute graph modularity from a partition {node -> community_id}.
    """
    # invert mapping: community_id -> set(nodes)
    inv = {}
    for u, c in partition.items():
        inv.setdefault(c, set()).add(u)
    comms = list(inv.values())
    return float(nx.algorithms.community.quality.modularity(G, comms, weight="weight"))
