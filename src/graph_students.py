
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from networkx.algorithms.community import girvan_newman
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try: # try Louvain if available 
    import community as community_louvain 
    HAVE_LOUVAIN = True
except Exception:
    HAVE_LOUVAIN = False

try:
    import igraph as ig  # for Infomap
    HAVE_IGRAPH = True
except Exception:
    HAVE_IGRAPH = False


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

# ------------------ TOOLS ------------------ 

def _communities_dict_from_sets(communities_sets):
    """
    Convertit une liste/iterable de sets de nœuds en dict {node -> community_id}
    """
    part = {}
    for cid, nodes in enumerate(communities_sets):
        for u in nodes:
            part[u] = cid
    return part


def compute_modularity(G, partition_dict):
    """
    Modularity à partir d'un dict {node -> community_id}.
    """
    inv = {}
    for u, c in partition_dict.items():
        inv.setdefault(c, set()).add(u)
    comms = list(inv.values())
    return float(nx.algorithms.community.quality.modularity(G, comms, weight="weight"))

def evaluate_partition(G, part_dict):
    """
    Petit résumé pratique pour comparer les méthodes.
    """
    mod = compute_modularity(G, part_dict)
    n_comm = len(set(part_dict.values()))
    sizes = pd.Series(part_dict).value_counts().sort_values(ascending=False)
    return {
        "modularity": float(mod),
        "n_communities": int(n_comm),
        "sizes": sizes.to_dict()
    }

def compute_purity(part, labels: pd.Series):
    """part: dict node->community ; labels: Series index=node -> class label"""
    groups = defaultdict(list)
    for node, comm in part.items():
        if node in labels.index:
            groups[comm].append(labels.loc[node])
    per_comm = {}
    num, den = 0, 0
    for c, vals in groups.items():
        if not vals: 
            per_comm[c] = np.nan
            continue
        cnt = Counter(vals)
        m = max(cnt.values())
        per_comm[c] = m / len(vals)
        num += m
        den += len(vals)
    overall = num / den if den else np.nan
    sizes = {c: len(v) for c, v in groups.items()}
    return overall, per_comm, sizes


# ---------- Méthodes de communautés ----------

def detect_communities_louvain(G):
    if not HAVE_LOUVAIN:
        raise RuntimeError("Louvain not available (install python-louvain).")
    part = community_louvain.best_partition(G, weight="weight", random_state=42)
    return part

def detect_communities_greedy(G):
    comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    return _communities_dict_from_sets(comms)


def detect_communities_girvan_newman(G, max_levels=4):
    """
    Girvan–Newman hiérarchique : on calcule plusieurs partitions successives et
    on choisit celle qui maximise la modularité parmi les premiers niveaux.
    """
    best_part = None
    best_mod = -1.0
    comp_gen = girvan_newman(G)

    # On évalue les 'max_levels' premières partitions
    for level, communities in zip(range(max_levels), comp_gen):
        comms = tuple(sorted(c) for c in communities)
        mod = nx.algorithms.community.quality.modularity(G, comms, weight="weight")
        if mod > best_mod:
            best_mod = mod
            best_part = _communities_dict_from_sets(comms)
    return best_part

def detect_communities_infomap(G):
    if not HAVE_IGRAPH:
        raise RuntimeError("Infomap not available (install igraph).")
    # Conversion NetworkX -> igraph
    G_ig = ig.Graph.from_networkx(G)
    # Infomap
    res = G_ig.community_infomap(edge_weights="weight")  # si poids absents, enlève l’arg
    # res is a VertexClustering; construire dict
    comms = [set(G_ig.vs[idx]["_nx_name"] for idx in cluster) for cluster in res]
    return _communities_dict_from_sets(comms)

def detect_communities(G, method="auto"):
    """
    Interface unifiée.
    method in {"auto", "louvain", "greedy", "girvan_newman", "infomap"}
    - auto: Louvain si dispo, sinon greedy
    """
    method = method.lower()
    if method == "auto":
        if HAVE_LOUVAIN:
            return detect_communities_louvain(G)
        return detect_communities_greedy(G)
    elif method == "louvain":
        return detect_communities_louvain(G)
    elif method == "greedy":
        return detect_communities_greedy(G)
    elif method == "girvan_newman":
        return detect_communities_girvan_newman(G)
    elif method == "infomap":
        return detect_communities_infomap(G)
    else:
        raise ValueError(f"Unknown method: {method}")



### fonction utilitaire 

def eval_partition(G, part_dict, labels):
    """Return metrics dict: modularity, n_comms, purity, NMI, ARI."""
    ev = evaluate_partition(G, part_dict)
    nodes = list(set(G.nodes()).intersection(labels.index))
    y_true = labels.loc[nodes].values
    y_pred = pd.Series(part_dict).reindex(nodes).values
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    purity, _, _ = compute_purity(part_dict, labels)
    return dict(
        modularity=float(ev["modularity"]),
        n_communities=int(ev["n_communities"]),
        purity=float(purity),
        NMI=float(nmi),
        ARI=float(ari)
    )

