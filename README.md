# **OULAD Network Exploration**  
*Network analysis applied to the OULAD dataset: community detection and bipartite modeling of learning interactions.*

Run
``` 
python -m pip install -e .
```
---

## **About the Project**

This project is part of an exploratory study based on the **OULAD (Open University Learning Analytics Dataset)**, which contains interactions from thousands of students enrolled at the Open University (UK).

The main goal is to leverage **graph theory** and **complex network analysis** to better understand **learning dynamics**, **interaction structures**, and **engagement behaviors**.

Two complementary approaches are explored:

###  **Approach 1 — Student–Student Network: Community Detection and Pattern Mining**
- Construction of a student interaction graph.  
- Application of several community detection algorithms (Louvain, Infomap, Label Propagation…).  
- Evaluation of modularity and topological properties (degree, centrality, density, clustering).  
- Behavioral interpretation of communities through pattern mining.  

###  **Approach 2 — Student–Resource Bipartite Network: Modeling and Exploration**
- Construction of a bipartite graph between students and learning resources from VLE activity logs.  
- Study of connectivity and projections onto student or resource subnetworks.  
- Detection of mixed communities and identification of learning profiles.  
- Comparative analysis between community profiles and success indicators.  




