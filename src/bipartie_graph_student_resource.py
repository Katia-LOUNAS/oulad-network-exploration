from turtle import pd
import networkx as nx



def build_student_activity_network(data, module_code='BBB', presentation='2013J', 
                                   min_clicks=5, top_students=None):
    """
    Construit un réseau bipartite ÉTUDIANT - TYPE DE RESSOURCE
    
    Plus simple et plus puissant que ressource individuelle !
    
    Paramètres:
    - module_code: code du cours 
    - presentation: session 
    - min_clicks: seuil minimum d'interactions
    - top_students: limiter le nombre d'étudiants (None = tous)
    """
    
    print("Filtrage des données...")
    
    # Obtenir les étudiants du module
    students_in_module = data['studentInfo'][
        (data['studentInfo']['code_module'] == module_code) & 
        (data['studentInfo']['code_presentation'] == presentation)
    ]
    print(f"   → {len(students_in_module)} étudiants trouvés")
    
    # Obtenir leurs interactions VLE
    interactions = data['studentVle'].merge(
        students_in_module[['id_student', 'final_result', 'gender', 'age_band']], 
        on='id_student'
    )
    
    # Ajouter le TYPE de ressource (pas la ressource elle-même!)
    interactions = interactions.merge(
        data['vle'][['id_site', 'activity_type']], 
        on='id_site'
    )
    print(f"   → {len(interactions)} interactions brutes")
    
    print("Agrégation par TYPE de ressource...")
    
    # CLEF: Agréger par étudiant ET type d'activité
    activity_summary = interactions.groupby(['id_student', 'activity_type']).agg({
        'sum_click': 'sum',
        'final_result': 'first',
        'gender': 'first',
        'age_band': 'first'
    }).reset_index()
    
    # Filtrer par seuil
    activity_summary = activity_summary[activity_summary['sum_click'] >= min_clicks]
    print(f"   → {len(activity_summary)} liens étudiant-type après filtrage")
    
    # Limiter aux étudiants les plus actifs si demandé
    if top_students:
        top_student_ids = activity_summary.groupby('id_student')['sum_click'].sum()\
            .nlargest(top_students).index
        activity_summary = activity_summary[activity_summary['id_student'].isin(top_student_ids)]
        print(f"   → Limité aux {top_students} étudiants les plus actifs")
    
    print("Construction du graphe...")
    
    # Créer le réseau
    G = nx.Graph()
    
    # Ajouter les ÉTUDIANTS comme nœuds
    students = activity_summary['id_student'].unique()
    for student in students:
        student_info = activity_summary[activity_summary['id_student'] == student].iloc[0]
        G.add_node(f"Student_{student}", 
                   bipartite=0,
                   node_type='student',
                   final_result=student_info['final_result'],
                   gender=student_info['gender'],
                   age_band=student_info['age_band'])
    
    # Ajouter les TYPES D'ACTIVITÉ comme nœuds
    activity_types = activity_summary['activity_type'].unique()
    for activity in activity_types:
        G.add_node(f"Activity_{activity}", 
                   bipartite=1,
                   node_type='activity',
                   activity_type=activity)
    
    print(f"   → {len(students)} étudiants + {len(activity_types)} types d'activités")
    
    # Ajouter les ARÊTES (pondérées par nombre de clics)
    for _, row in activity_summary.iterrows():
        G.add_edge(f"Student_{row['id_student']}", 
                   f"Activity_{row['activity_type']}", 
                   weight=row['sum_click'])
    
    print(f"   → {G.number_of_edges()} arêtes créées")
    print(f"Réseau construit avec succès!\n")
    
    return G, activity_summary



def export_to_gephi(G, filename_prefix='student_network'):
    """
    Exporte le réseau au format GEXF pour Gephi
    Gephi peut lire ce format directement avec tous les attributs!
    """
    
    import pandas as pandas_lib  # Import local pour éviter les conflits
    
    print(f"💾 Export pour Gephi...")
    
    # 1. Export GEXF (format recommandé pour Gephi)
    gexf_file = f"{filename_prefix}.gexf"
    nx.write_gexf(G, gexf_file)
    print(f"   ✅ Fichier GEXF créé: {gexf_file}")
    
    # 2. Export GraphML (alternative)
    graphml_file = f"{filename_prefix}.graphml"
    nx.write_graphml(G, graphml_file)
    print(f"   ✅ Fichier GraphML créé: {graphml_file}")
    
    # 3. Export CSV des arêtes (pour import manuel)
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'Source': u,
            'Target': v,
            'Weight': data.get('weight', 1),
            'Type': 'Undirected'
        })
    
    edges_df = pandas_lib.DataFrame(edges_data)
    edges_csv = f"{filename_prefix}_edges.csv"
    edges_df.to_csv(edges_csv, index=False)
    print(f"   ✅ Fichier CSV arêtes créé: {edges_csv}")
    
    # 4. Export CSV des nœuds (avec tous les attributs)
    nodes_data = []
    for node, attrs in G.nodes(data=True):
        node_info = {'Id': node, 'Label': node}
        node_info.update(attrs)
        nodes_data.append(node_info)
    
    nodes_df = pandas_lib.DataFrame(nodes_data)
    nodes_csv = f"{filename_prefix}_nodes.csv"
    nodes_df.to_csv(nodes_csv, index=False)
    print(f"   ✅ Fichier CSV nœuds créé: {nodes_csv}")
    
    print(f"\n📊 Pour ouvrir dans Gephi:")
    print(f"   1. Ouvrir Gephi")
    print(f"   2. File → Open → Sélectionner '{gexf_file}'")
    print(f"   3. Dans 'Appearance', colorier par 'final_result' ou 'node_type'")
    print(f"   4. Appliquer un layout (ForceAtlas2 recommandé)")
    print(f"   5. Calculer les statistiques réseau (Tools → Statistics)\n")
    
    return {
        'gexf': gexf_file,
        'graphml': graphml_file,
        'edges_csv': edges_csv,
        'nodes_csv': nodes_csv
    }

def export_with_layout(G, filename_prefix='student_network'):
    """
    Exporte avec un layout pré-calculé pour Gephi
    Utile pour les réseaux bipartites!
    """
    
    print(f"💾 Export avec layout bipartite...")
    
    # Calculer un layout bipartite
    students = {n for n, d in G.nodes(data=True) if d['node_type'] == 'student'}
    activities = {n for n, d in G.nodes(data=True) if d['node_type'] == 'activity'}
    
    # Positions
    pos = {}
    # Étudiants à gauche (colonne verticale)
    for idx, node in enumerate(students):
        pos[node] = (0, idx * 10)
    
    # Activités à droite (colonne verticale)
    for idx, node in enumerate(activities):
        pos[node] = (100, idx * 50)
    
    # Ajouter les positions comme attributs
    for node, (x, y) in pos.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y
        G.nodes[node]['z'] = 0
    
    # Exporter avec positions
    gexf_file = f"{filename_prefix}_with_layout.gexf"
    nx.write_gexf(G, gexf_file)
    print(f"   ✅ Fichier avec layout créé: {gexf_file}")
    print(f"   → Gephi gardera ce layout bipartite!\n")
    
    return gexf_file



