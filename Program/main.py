import tkinter as tk
from tkinter import ttk
from collections import deque
import tkinter.simpledialog

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        self.graph.setdefault(u, []).append(v)
        self.graph.setdefault(v, []).append(u)

class Node:
    def __init__(self, node_id, x, y, circle_id, text_id):
        self.id = node_id
        self.x = x
        self.y = y
        self.circle_id = circle_id
        self.text_id = text_id

class UnionFind:
    def __init__(self, size):
        # Inițializăm părinții și rank-ul fiecărui nod
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node):
        # Găsim reprezentantul (părintele) mulțimii pentru nodul dat
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])  # Path compression
        return self.parent[node]

    def union(self, u, v):
        # Unim două mulțimi bazat pe rank
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

class GraphWeighted:
    def __init__(self, is_directed=False):
        self.edges = []  # Listă de muchii: (cost, nod1, nod2)
        self.adjacency_list = {}  # Lista de adiacență
        self.nodes = set()  # Set pentru stocarea nodurilor
        self.is_directed = is_directed  # Specificăm dacă graful este orientat

    def add_edge(self, u, v, weight, cost=0):
        # Adăugăm muchia directă
        self.edges.append((weight, u, v))
        self.nodes.update([u, v])  # Adăugăm nodurile în set

        # Actualizăm lista de adiacență
        if u not in self.adjacency_list:
            self.adjacency_list[u] = []
        self.adjacency_list[u].append((v, weight, cost))

        # Adaugă muchia inversă cu capacitate 0 și cost negativ
        if v not in self.adjacency_list:
            self.adjacency_list[v] = []
        self.adjacency_list[v].append((u, 0, -cost))  # Capacitate inversă 0, cost negativ
        
        '''# Dacă graful nu este orientat, adăugăm muchia și invers ca muchie normala
        if not self.is_directed:  
            if v not in self.adjacency_list:
                self.adjacency_list[v] = []
            self.adjacency_list[v].append((u, weight, cost))
            if u not in self.adjacency_list:
                self.adjacency_list[u] = []
            self.adjacency_list[u].append((v, weight, cost))'''

    def get_neighbors(self, node):
        # Returnăm vecinii unui nod din lista de adiacență
        return self.adjacency_list.get(node, [])

    def kruskal_mst(self):
        # Sortăm muchiile după cost
        self.edges.sort()
        parent = {node: node for node in self.nodes}  # Părinții pentru Union-Find
        rank = {node: 0 for node in self.nodes}  # Rank-ul pentru Union-Find

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                elif rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1

        mst = []
        total_cost = 0

        for weight, u, v in self.edges:
            if find(u) != find(v):
                union(u, v)
                mst.append((u, v, weight))
                total_cost += weight

        return mst, total_cost

    def boruvka_mst(self):
        parent = {node: node for node in self.nodes}
        rank = {node: 0 for node in self.nodes}

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                elif rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1

        mst = []
        total_cost = 0
        num_components = len(self.nodes)

        while num_components > 1:
            cheapest = {node: None for node in self.nodes}

            # Găsim cea mai ieftină muchie pentru fiecare componentă
            for weight, u, v in self.edges:
                root_u = find(u)
                root_v = find(v)
                if root_u != root_v:
                    if cheapest[root_u] is None or weight < cheapest[root_u][0]:
                        cheapest[root_u] = (weight, u, v)
                    if cheapest[root_v] is None or weight < cheapest[root_v][0]:
                        cheapest[root_v] = (weight, u, v)

            # Adăugăm muchiile selectate în MST
            for node, edge in cheapest.items():
                if edge is not None:
                    weight, u, v = edge
                    root_u = find(u)
                    root_v = find(v)
                    if root_u != root_v:
                        union(u, v)
                        mst.append((u, v, weight))
                        total_cost += weight
                        num_components -= 1

        return mst, total_cost

    # Metoda pentru a furniza o lista simpla de adiacenta pt algoritmii ce nu au nevoie de cost
    def get_unweighted_adjacency_list(self):
        """Returnează o listă de adiacență simplă (fără ponderi)."""
        adjacency_list = {}
        for u, neighbors in self.adjacency_list.items():
            adjacency_list[u] = [v for v, _, _ in neighbors]  # Ignorăm ponderile și costurile
        return adjacency_list

class Edge:
    def __init__(self, from_node_id, to_node_id, line_id):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.line_id = line_id


class FlowNetwork:
    def __init__(self, graph):
        self.graph = {u: {} for u in graph.nodes}  # Residual graph
        self.flow = {}  # Current flow
        self.cost = {}  # Edge costs for min cost flow

        # Initialize residual graph from GraphWeighted
        for u in graph.nodes:
            for v, weight, cost in graph.adjacency_list.get(u, []):
                self.graph[u][v] = weight
                self.flow[u, v] = 0
                self.cost[u, v] = cost
                if v not in self.graph:
                    self.graph[v] = {}
                if u not in self.graph[v]:
                    self.graph[v][u] = 0
                self.flow[v, u] = 0
                self.cost[v, u] = -cost

    def reset_flow(self):
        """Resetează fluxurile curente la 0 și reinițializează capacitățile reziduale."""
        for u in self.graph:
            for v in self.graph[u]:
                self.flow[u, v] = 0
                self.flow[v, u] = 0
                self.graph[u][v] = max(self.graph[u][v], 0)  # Resetăm la capacitatea inițială

    def find_path_dfs(self, source, sink, path):
        if source == sink:
            return path
        
        for next_node in self.graph[source]:
            residual = self.graph[source][next_node] - self.flow.get((source, next_node), 0)
            if residual > 0 and (source, next_node) not in path:
                result = self.find_path_dfs(next_node, sink, path + [(source, next_node)])
                if result != None:
                    return result
        return None

    def ford_fulkerson(self, source, sink, use_bfs=False):
        """Aplica Ford-Fulkerson cu DFS sau Edmonds-Karp cu BFS."""
        if use_bfs:
            return self.edmonds_karp(source, sink)  # Trebuie să implementezi BFS dacă nu există deja
        else:
            return self.ford_fulkerson_initial(source, sink)
    
    def edmonds_karp(self, source, sink):
        """Implementarea Edmonds-Karp pentru flux maxim folosind BFS."""
        max_flow = 0
    
        while True:
            path = self.find_path_bfs(source, sink)
            if not path:
                break  # Nu mai există drumuri augmentante
            
            # Determinăm capacitatea minimă de pe acest drum (bottleneck)
            bottleneck = min(self.graph[u][v] - self.flow[u, v] for u, v in path)
    
            # Actualizăm fluxurile
            for u, v in path:
                self.flow[u, v] += bottleneck
                self.flow[v, u] -= bottleneck
    
            max_flow += bottleneck
    
        return max_flow

    def find_path_bfs(self, source, sink):
        visited = {source: None}
        queue = deque([source])

        print(f"Pornim BFS de la sursa {source} la destinația {sink}.")
        while queue:
            u = queue.popleft()
            print(f"Procesăm nodul {u}.")
            
            for v in self.graph[u]:
                residual = self.graph[u][v] - self.flow[u, v]
                print(f"Verificăm muchia ({u} -> {v}): Capacitate reziduală = {residual}.")
                if residual > 0 and v not in visited:
                    visited[v] = u
                    queue.append(v)
                    print(f"Adăugăm nodul {v} la coadă.")
                    if v == sink:
                        print(f"Destinația {sink} a fost atinsă.")
                        break

        if sink not in visited:
            print("Nu există drum de mărire.")
            return None

        # Reconstruct path
        path = []
        current = sink
        while current != source:
            prev = visited[current]
            path.append((prev, current))
            current = prev
        path.reverse()

        print(f"Drum reconstruit: {path}")
        return path

    def ford_fulkerson_initial(self, source, sink):
        """Aplică Ford-Fulkerson pentru a obține flux maxim inițial."""
        def find_path(source, sink, path):
            if source == sink:
                return path
            for v in self.graph[source]:
                residual = self.graph[source][v] - self.flow[source, v]
                if residual > 0 and (source, v) not in path:
                    result = find_path(v, sink, path + [(source, v)])
                    if result:
                        return result
            return None

        max_flow = 0
        path = find_path(source, sink, [])
        while path:
            bottleneck = min(self.graph[u][v] - self.flow[u, v] for u, v in path)
            for u, v in path:
                self.flow[u, v] += bottleneck
                self.flow[v, u] -= bottleneck
            max_flow += bottleneck
            path = find_path(source, sink, [])
        return max_flow

    def bellman_ford(self, source):
        """Detectează cicluri negative."""
        distance = {node: float('inf') for node in self.graph}
        predecessor = {node: None for node in self.graph}
        distance[source] = 0

        for _ in range(len(self.graph) - 1):
            for u in self.graph:
                for v in self.graph[u]:
                    residual = self.graph[u][v] - self.flow[u, v]
                    if residual > 0 and distance[u] + self.cost[u, v] < distance[v]:
                        distance[v] = distance[u] + self.cost[u, v]
                        predecessor[v] = u

        for u in self.graph:
            for v in self.graph[u]:
                residual = self.graph[u][v] - self.flow[u, v]
                if residual > 0 and distance[u] + self.cost[u, v] < distance[v]:
                    return True, self.find_negative_cycle(u, predecessor)
        return False, None

    def find_negative_cycle(self, start, predecessor):
        """Găsește un ciclu negativ."""
        visited = set()
        current = start
        while current not in visited:
            visited.add(current)
            current = predecessor[current]
        cycle = []
        cycle_start = current
        current = predecessor[current]
        cycle.append((current, cycle_start))
        while current != cycle_start:
            next_node = predecessor[current]
            cycle.append((next_node, current))
            current = next_node
        return cycle

    def cycle_canceling(self, source, sink):
        """Aplică algoritmul Cycle-Canceling pentru a minimiza costul fluxului."""
        print("Start Cycle-Canceling")

        # Pas 1: Flux maxim inițial
        initial_flow = self.ford_fulkerson_initial(source, sink)
        print(f"Initial max flow: {initial_flow}")

        # Pas 2: Eliminare cicluri negative
        while True:
            has_negative_cycle, cycle = self.bellman_ford(source)
            if not has_negative_cycle:
                break

            print(f"Negative cycle found: {cycle}")

            bottleneck = min(self.graph[u][v] - self.flow[u, v] for u, v in cycle)
            print(f"Bottleneck: {bottleneck}")

            for u, v in cycle:
                self.flow[u, v] += bottleneck
                self.flow[v, u] -= bottleneck
                print(f"Updated flow: {u} -> {v}: {self.flow[u, v]}, {v} -> {u}: {self.flow[v, u]}")

        # Pas 3: Verificare finală cu Bellman-Ford
        has_negative_cycle, _ = self.bellman_ford(source)
        if has_negative_cycle:
            print("⚠️ Atenție! Există încă cicluri negative după Cycle-Canceling, fluxul poate să nu fie optim!")
        else:
            print("✅ Verificare completă: Nu mai există cicluri negative, fluxul este minimizat corect.")
        
        # Pas 4: Calcul cost minim final
        max_flow = sum(self.flow[u, v] for u, v in self.flow if v == sink)
        total_cost = sum(
            self.flow[u, v] * self.cost[u, v]
            for u, v in self.flow if self.flow[u, v] > 0
        )

        print(f"Final max flow: {max_flow}, Final min cost: {total_cost}")
        return max_flow, total_cost


    def validate_min_cost_flow(self):
        """Verifică dacă fluxul minimizează costurile pentru fiecare drum augmentat."""
        total_cost = 0
        for u in self.graph:
            for v in self.graph[u]:
                if self.flow[u, v] > 0:
                    cost_for_edge = self.flow[u, v] * self.cost[u, v]
                    total_cost += cost_for_edge
                    print(f"Flux pe muchia ({u} -> {v}): {self.flow[u, v]}, Cost: {self.cost[u, v]}, Cost Total: {cost_for_edge}")
        
        print(f"Cost total calculat: {total_cost}")
        return total_cost

    def scaling_ford_fulkerson(self, source, sink):
        # Determine the maximum capacity in the graph
        max_capacity = max(
            weight for u in self.graph for v, weight in self.graph[u].items()
        )

        # Start with the highest power of 2 <= max_capacity
        delta = 1
        while delta <= max_capacity:
            delta *= 2
        delta //= 2

        max_flow = 0

        while delta > 0:
            print(f"Delta actual: {delta}")
            path = self.find_scaling_path(source, sink, delta)

            while path is not None:
                print(f"Drum găsit pentru delta {delta}: {path}")

                # Find minimum residual capacity along the path
                flow = float('inf')
                for u, v in path:
                    residual = self.graph[u][v] - self.flow[u, v]
                    flow = min(flow, residual)

                print(f"Bottleneck găsit: {flow}")

                # Update flows
                for u, v in path:
                    self.flow[u, v] += flow
                    self.flow[v, u] -= flow

                max_flow += flow
                print(f"Flux curent după actualizare: {self.flow}")

                # Find another augmenting path with the same delta
                path = self.find_scaling_path(source, sink, delta)

            # Decrease delta
            delta //= 2

        print(f"Flux maxim cu scaling: {max_flow}")
        return max_flow

    def find_scaling_path(self, source, sink, delta):
        visited = {source: None}
        queue = deque([source])

        while queue:
            u = queue.popleft()

            for v in self.graph[u]:
                residual = self.graph[u][v] - self.flow[u, v]
                if residual >= delta and v not in visited:
                    visited[v] = u
                    queue.append(v)
                    if v == sink:
                        break

        if sink not in visited:
            return None

        # Reconstruct path
        path = []
        current = sink
        while current != source:
            prev = visited[current]
            path.append((prev, current))
            current = prev
        path.reverse()

        return path
    

class GraphGUI:
    def color_component_dfs_transposed(self, node_id, visited, component_id, transposed_graph):
        stack = [node_id]
        visited[node_id] = True
        color = self.get_component_color(component_id)

        while stack:
            current_node = stack.pop()
            self.canvas.itemconfig(self.nodes[current_node].circle_id, fill=color)
            self.master.update()
            self.master.after(500)  # Pause for visualization

            # Traverse the transpose graph
            for neighbor, _, _ in transposed_graph.get(current_node, []):  # Ignore weights and costs
                if neighbor not in visited:
                    visited[neighbor] = True
                    stack.append(neighbor)

    def transpose_graph(self):
        transposed_graph = {}
        for u, neighbors in self.graph.adjacency_list.items():
            for v, weight, cost in neighbors:  # Ignorăm ponderile si costurile
                transposed_graph.setdefault(v, []).append((u, weight, cost))  # Reverse the edge        
        return transposed_graph

    def get_component_color(self, component_id):
        colors = ["lightgreen", "lightcoral", "lightblue", "yellow", "pink", "lightgrey", "orange"]
        return colors[component_id % len(colors)]

    def dfs_fill_stack(self, node_id, visited, stack, adjacency_list):
        visited[node_id] = True
    
        for neighbor in adjacency_list.get(node_id, []):
            if neighbor not in visited:
                self.dfs_fill_stack(neighbor, visited, stack, adjacency_list)
    
        stack.append(node_id)  # Add node to stack after all its neighbors are processed

    def __init__(self, master):
        self.master = master
        master.title("Graph Algorithms Visualization")

        # Flags pentru tipurile de graf
        self.is_directed = tk.BooleanVar(value=False)
        self.is_weighted = tk.BooleanVar(value=True)

        # Modes
        self.mode = tk.StringVar()
        self.mode.set("draw_node")

        # Algorithm selection
        self.algorithm = tk.StringVar()
        self.algorithm.set("Recursive DFS")

        # Starting node selection
        self.start_node = tk.IntVar()
        
        # Ending node selection
        self.end_node = tk.IntVar()

        # Node counter
        self.node_counter = 0

        # Data structures
        self.nodes = {}  # node_id: Node
        self.edges = []  # list of Edge
        self.graph = GraphWeighted(is_directed=self.is_directed.get())  # Graful folosește is_directed

        # Create GUI components
        self.create_widgets()

    def transform_to_tree_dfs(self, start_node_id):
        visited = set()
        tree_edges = []  # Muchiile arborelui rezultat

        def dfs(node_id):
            visited.add(node_id)
            for neighbor, _, _ in self.graph.adjacency_list.get(node_id, []):  # Ignorăm ponderile si costurile
                if neighbor not in visited:
                    # Adăugăm muchia la arbore și o colorăm
                    self.highlight_tree_edge(node_id, neighbor)
                    tree_edges.append((node_id, neighbor))
                    dfs(neighbor)

        dfs(start_node_id)
        print("Muchiile arborelui DFS:", tree_edges)

    def highlight_tree_edge(self, from_node_id, to_node_id):
        for edge in self.edges:
            if (edge.from_node_id == from_node_id and edge.to_node_id == to_node_id) or \
                (not self.is_directed.get() and edge.from_node_id == to_node_id and edge.to_node_id == from_node_id):
                 self.canvas.itemconfig(edge.line_id, fill="green", width=2)
                 self.master.update()
                 self.master.after(250)
                 return

    def draw_edge(self, from_node_id, to_node_id, weight=None, cost=None):
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        arrow = "last" if self.is_directed.get() else None
        line_id = self.canvas.create_line(from_node.x, from_node.y, to_node.x, to_node.y, arrow=arrow)
        text_x = (from_node.x + to_node.x) / 2
        text_y = (from_node.y + to_node.y) / 2

        # Afișează textul ponderii și costului, dacă sunt specificate
        text_label = ""
        if weight is not None:
            text_label += f"{weight}"
        if cost is not None:
            text_label += f", {cost}" if text_label else f"{cost}"
        if text_label:
            self.canvas.create_text(text_x, text_y - 10, text=text_label)

        # Salvează muchia desenată
        edge = Edge(from_node_id, to_node_id, line_id)
        self.edges.append(edge)
    
    def create_widgets(self):
        # Main frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas
        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create controls
        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Graph type selection
        graph_type_frame = tk.LabelFrame(control_frame, text="Graph Type")
        graph_type_frame.pack(pady=10)

        tk.Checkbutton(graph_type_frame, text="Directed", variable=self.is_directed,
                       command=self.update_graph_type).pack(anchor=tk.W)

        # Graph weighting type
        weighting_frame = tk.LabelFrame(control_frame, text="Graph Weighting")
        weighting_frame.pack(pady=10)

        self.weighting_type = tk.StringVar(value="Simple")
        tk.Radiobutton(weighting_frame, text="Simple", variable=self.weighting_type, value="Simple").pack(anchor=tk.W)
        tk.Radiobutton(weighting_frame, text="Weighted", variable=self.weighting_type, value="Weighted").pack(anchor=tk.W)
        tk.Radiobutton(weighting_frame, text="Weighted + Costed", variable=self.weighting_type, value="Weighted+Costed").pack(anchor=tk.W)

        # Modes
        modes_frame = tk.LabelFrame(control_frame, text="Mode")
        modes_frame.pack(pady=10)

        tk.Radiobutton(modes_frame, text="Draw Nodes", variable=self.mode, value="draw_node").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Draw Edges", variable=self.mode, value="draw_edge").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Move Nodes", variable=self.mode, value="move_node").pack(anchor=tk.W)

        # Algorithm selection
        alg_frame = tk.LabelFrame(control_frame, text="Algorithm")
        alg_frame.pack(pady=10)

        algorithms = [
            "Recursive DFS",
            "DFS",
            "BFS",
            "Connected Components",
            "Strongly Connected Components",
            "Topological Sort",
            "Graph Centers",
            "Transform to DFS Tree",
            "Eager Prim's MST",
            "Kruskal's MST",
            "Borůvka's MST",
            "Ford-Fulkerson (DFS)",
            "Edmonds-Karp (BFS)",
            "Min Cost Flow"
        ]
        alg_menu = ttk.OptionMenu(alg_frame, self.algorithm, self.algorithm.get(), *algorithms)
        alg_menu.pack()

        # Starting node selection
        start_node_frame = tk.LabelFrame(control_frame, text="Start Node")
        start_node_frame.pack(pady=10)

        self.start_node_menu = ttk.OptionMenu(start_node_frame, self.start_node, None)
        self.start_node_menu.pack()

        # Ending node selection
        end_node_frame = tk.LabelFrame(control_frame, text="End Node")
        end_node_frame.pack(pady=10)

        self.end_node_menu = ttk.OptionMenu(end_node_frame, self.end_node, None)
        self.end_node_menu.pack()

        # Highlight Start Node button
        highlight_button = tk.Button(control_frame, text="Highlight Start Node", command=self.highlight_start_node)
        highlight_button.pack(pady=10)

        # Highlight End Node button
        highlight_end_button = tk.Button(control_frame, text="Highlight End Node", command=self.highlight_end_node)
        highlight_end_button.pack(pady=10)

        # Run button
        run_button = tk.Button(control_frame, text="Run", command=self.run_algorithm)
        run_button.pack(pady=10)

        # Clear Canvas button
        clear_button = tk.Button(control_frame, text="Clear Canvas", command=self.clear_canvas)
        clear_button.pack(pady=10)

        # Bind events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # For moving nodes
        self.selected_node_id = None
        self.dragging = False

        # For edge drawing
        self.edge_start_node = None

    def update_graph_type(self):
        self.graph = GraphWeighted(is_directed=self.is_directed.get())
        self.clear_canvas()  # Resetează nodurile și muchiile

    def update_start_node_menu(self):
        menu = self.start_node_menu["menu"]
        menu.delete(0, "end")
        for node_id in self.nodes.keys():
            menu.add_command(label=str(node_id), command=lambda value=node_id: self.start_node.set(value))
        if self.nodes:
            self.start_node.set(next(iter(self.nodes)))
        else:
            self.start_node.set(None)

    def update_end_node_menu(self):
        menu = self.end_node_menu["menu"]
        menu.delete(0, "end")
        for node_id in self.nodes.keys():
            menu.add_command(label=str(node_id), command=lambda value=node_id: self.end_node.set(value))
        if self.nodes:
            self.end_node.set(next(iter(self.nodes)))
        else:
            self.end_node.set(None)
        
    # Visual Feedback la Alegerea Nodului de Start
    def highlight_start_node(self):
        start_node_id = self.start_node.get()
        for node in self.nodes.values():
            color = "lime" if node.id == start_node_id else "lightblue"
            self.canvas.itemconfig(node.circle_id, fill=color)
        print(f"Nodul de start este {start_node_id}.")
        
    def highlight_end_node(self):
        end_node_id = self.end_node.get()
        for node in self.nodes.values():
            color = "red" if node.id == end_node_id else "lightblue"
            self.canvas.itemconfig(node.circle_id, fill=color)
        print(f"Nodul de final este {end_node_id}.")
        
    def canvas_click(self, event):
        if self.mode.get() == "draw_node":
            self.add_node(event.x, event.y)
        elif self.mode.get() == "draw_edge":
            self.select_edge_node(event.x, event.y)
        elif self.mode.get() == "move_node":
            self.start_move_node(event.x, event.y)

    def add_node(self, x, y):
        node_id = self.node_counter
        self.node_counter += 1

        r = 20  # radius
        circle_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="lightblue")
        text_id = self.canvas.create_text(x, y, text=str(node_id))

        node = Node(node_id, x, y, circle_id, text_id)
        self.nodes[node_id] = node

        # Update start and end node menus
        self.update_start_node_menu()
        self.update_end_node_menu()


    def select_edge_node(self, x, y):
        node_id = self.get_node_at_position(x, y)
        if node_id is not None:
            if self.edge_start_node is None:
                self.edge_start_node = node_id
                # Highlight the node
                self.canvas.itemconfig(self.nodes[node_id].circle_id, outline="red", width=2)
            else:
                from_node_id = self.edge_start_node
                to_node_id = node_id
                self.add_edge(from_node_id, to_node_id)
                # Remove highlight
                self.canvas.itemconfig(self.nodes[from_node_id].circle_id, outline="black", width=1)
                self.edge_start_node = None

    def get_node_at_position(self, x, y):
        overlapping = self.canvas.find_overlapping(x - 1, y - 1, x + 1, y + 1)
        for item in overlapping:
            for node_id, node in self.nodes.items():
                if item == node.circle_id or item == node.text_id:
                    return node_id
        return None

    def add_edge(self, from_node_id, to_node_id):
        graph_type = self.weighting_type.get()

        if graph_type == "Simple":
            weight = 1
            cost = 0
        elif graph_type == "Weighted":
            weight_str = tk.simpledialog.askstring("Weight", "Introduceți ponderea muchiei:", parent=self.master)
            try:
                weight = int(weight_str)
                cost = 0
            except (ValueError, TypeError):
                print("Pondere invalidă. Muchia nu a fost adăugată.")
                return
        elif graph_type == "Weighted+Costed":
            weight_str = tk.simpledialog.askstring("Weight", "Introduceți ponderea muchiei:", parent=self.master)
            cost_str = tk.simpledialog.askstring("Cost", "Introduceți costul muchiei:", parent=self.master)
            try:
                weight = int(weight_str)
                cost = int(cost_str)
            except (ValueError, TypeError):
                print("Pondere sau cost invalid. Muchia nu a fost adăugată.")
                return

        # Adăugăm muchia în graf
        if not self.is_directed.get() and from_node_id > to_node_id:
            from_node_id, to_node_id = to_node_id, from_node_id

        self.graph.add_edge(from_node_id, to_node_id, weight, cost)
        self.draw_edge(from_node_id, to_node_id, weight, cost)
    
    def start_move_node(self, x, y):
        node_id = self.get_node_at_position(x, y)
        if node_id is not None:
            self.selected_node_id = node_id
            self.dragging = True

    def canvas_drag(self, event):
        if self.mode.get() == "move_node" and self.dragging and self.selected_node_id is not None:
            node = self.nodes[self.selected_node_id]
            dx = event.x - node.x
            dy = event.y - node.y
            self.canvas.move(node.circle_id, dx, dy)
            self.canvas.move(node.text_id, dx, dy)
            node.x = event.x
            node.y = event.y
            self.update_edges(self.selected_node_id)

    def canvas_release(self, event):
        if self.mode.get() == "move_node" and self.dragging:
            self.dragging = False
            self.selected_node_id = None

    def update_edges(self, node_id):
        for edge in self.edges:
            if edge.from_node_id == node_id or edge.to_node_id == node_id:
                from_node = self.nodes[edge.from_node_id]
                to_node = self.nodes[edge.to_node_id]
                self.canvas.coords(edge.line_id, from_node.x, from_node.y, to_node.x, to_node.y)

    def run_algorithm(self):
        print(f"Algoritmul selectat: {self.algorithm.get()}, Start Node: {self.start_node.get()}, End Node: {self.end_node.get()}")
        
        start_node_id = self.start_node.get()
        if start_node_id is None or start_node_id not in self.nodes:
            print("Nodul de start nu este valid.")
            return

        algorithm = self.algorithm.get()

        # Resetăm culorile nodurilor și muchiilor
        for node in self.nodes.values():
            self.canvas.itemconfig(node.circle_id, fill="lightblue")
        for edge in self.edges:
            self.canvas.itemconfig(edge.line_id, fill="black", width=1)

        # Selectăm algoritmul ales
        if algorithm == "Recursive DFS":
            visited = {}
            self.recursive_dfs(start_node_id, visited)
        elif algorithm == "DFS":
            self.dfs(start_node_id)
        elif algorithm == "BFS":
            self.bfs(start_node_id)
        elif algorithm == "Connected Components":
            self.color_connected_components()
        elif algorithm == "Strongly Connected Components":
            self.color_strongly_connected_components()
        elif algorithm == "Topological Sort":
            self.topological_sort()
        elif algorithm == "Graph Centers":
            self.find_graph_centers()
        elif algorithm == "Transform to DFS Tree":
            self.transform_to_tree_dfs(start_node_id)
        elif algorithm == "Eager Prim's MST":
            self.eager_prims_mst()
        elif algorithm == "Kruskal's MST":
            self.kruskal_mst()
        elif algorithm == "Borůvka's MST":
            print("Borůvka MST început...")
            self.boruvka_mst()
        elif algorithm == "Ford-Fulkerson (DFS)":
            flow_network = FlowNetwork(self.graph)
            max_flow = flow_network.ford_fulkerson(self.start_node.get(), self.end_node.get(), use_bfs=False)
            print(f"Flux maxim: {max_flow}")
            self.highlight_flow(flow_network.flow)
        elif algorithm == "Edmonds-Karp (BFS)":
            flow_network = FlowNetwork(self.graph)
            max_flow = flow_network.ford_fulkerson(self.start_node.get(), self.end_node.get(), use_bfs=True)
            print(f"Flux maxim: {max_flow}")
            self.highlight_flow(flow_network.flow)
        elif algorithm == "Min Cost Flow":
            print("Începem calculul pentru Min Cost Flow...")  # Debugging start

            # Inițializează rețeaua de flux
            flow_network = FlowNetwork(self.graph)
            flow_network.reset_flow()  # Asigură-te că fluxurile sunt resetate

            # Debugging pentru starea inițială
            print("Graful rezidual inițial:", flow_network.graph)
            print("Fluxurile inițiale:", flow_network.flow)

            # Rulează algoritmul
            max_flow, min_cost = flow_network.cycle_canceling(self.start_node.get(), self.end_node.get())

            # Debugging pentru rezultatele intermediare
            print(f"Flux maxim: {max_flow}, Cost minim: {min_cost}")
            print(f"Fluxul curent: {flow_network.flow}")

            # Evidențiază fluxul în interfața grafică
            self.highlight_flow(flow_network.flow)

    def recursive_dfs(self, node_id, visited):
        visited[node_id] = True
        # Highlight the node
        self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="yellow")
        self.master.update()
        self.master.after(500)  # Pause for visualization

        adjacency_list = self.graph.get_unweighted_adjacency_list()
        for neighbor in adjacency_list.get(node_id, []):
            if neighbor not in visited:
                # Optionally highlight the edge
                self.highlight_edge(node_id, neighbor)
                self.recursive_dfs(neighbor, visited)

    def dfs(self, start_node_id):
        stack = [start_node_id]
        visited = {}

        # Folosim lista de adiacență neponderată
        adjacency_list = self.graph.get_unweighted_adjacency_list()

        while stack:
            node_id = stack.pop()
            if node_id not in visited:
                visited[node_id] = True
                # Highlight the node
                self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="yellow")
                self.master.update()
                self.master.after(500)  # Pause for visualization

                for neighbor in reversed(adjacency_list.get(node_id, [])):
                    stack.append(neighbor)

    def bfs(self, start_node_id):
        visited = {start_node_id: True}
        queue = deque([start_node_id])

        # Folosim lista de adiacență neponderată
        adjacency_list = self.graph.get_unweighted_adjacency_list()

        while queue:
            node_id = queue.popleft()
            # Highlight the node
            self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="yellow")
            self.master.update()
            self.master.after(500)  # Pause for visualization

            for neighbor in adjacency_list.get(node_id, []):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited[neighbor] = True

    #colorarea componentelor conexe
    def color_connected_components(self):
        visited = {}
        component_id = 0

        # Folosim lista de adiacență neponderată
        adjacency_list = self.graph.get_unweighted_adjacency_list()

        for node_id in self.nodes.keys():
            if node_id not in visited:
                self.color_component_dfs(node_id, visited, component_id, adjacency_list)
                component_id += 1

    def color_component_dfs(self, node_id, visited, component_id, adjacency_list):
        stack = [node_id]
        visited[node_id] = True
        color = self.get_component_color(component_id)

        while stack:
            node_id = stack.pop()
            self.canvas.itemconfig(self.nodes[node_id].circle_id, fill=color)

            for neighbor in adjacency_list.get(node_id, []):
                if neighbor not in visited:
                    visited[neighbor] = True
                    stack.append(neighbor)

    #colorarea componentelor tare conexe
    def color_strongly_connected_components(self):
        # Step 1: Perform DFS to fill the stack with nodes in order of finish times
        visited = {}
        stack = []
        adjacency_list = self.graph.get_unweighted_adjacency_list()  # Use unweighted adjacency list

        for node_id in self.nodes.keys():
            if node_id not in visited:
                self.dfs_fill_stack(node_id, visited, stack, adjacency_list)

        # Step 2: Create the transpose graph
        transposed_graph = self.transpose_graph()

        # Step 3: Perform DFS on the transpose graph in the order of the stack
        visited.clear()
        component_id = 0
        while stack:
            node_id = stack.pop()
            if node_id not in visited:
                self.color_component_dfs_transposed(node_id, visited, component_id, transposed_graph)
                component_id += 1

    #sortare topologica
    def topological_sort(self):
        visited = {}
        stack = []

        adjacency_list = self.graph.get_unweighted_adjacency_list()

        for node_id in self.nodes.keys():
            if node_id not in visited:
                self.topological_sort_dfs(node_id, visited, stack, adjacency_list)

        while stack:
            node_id = stack.pop()
            self.canvas.itemconfig(self.nodes[node_id].circle_id,
                                   fill="orange")  # Optional: schimbă culoarea pentru a indica ordinea
            self.master.update()
            self.master.after(500)

    #pt topological_sort_dfs
    def topological_sort_dfs(self, node_id, visited, stack, adjacency_list):
        visited[node_id] = True

        for neighbor in adjacency_list.get(node_id, []):
            if neighbor not in visited:
                self.topological_sort_dfs(neighbor, visited, stack, adjacency_list)

        stack.append(node_id)

    def highlight_edge(self, from_node_id, to_node_id):
        for edge in self.edges:
            if (edge.from_node_id == from_node_id and edge.to_node_id == to_node_id) or \
               (edge.from_node_id == to_node_id and edge.to_node_id == from_node_id):
                self.canvas.itemconfig(edge.line_id, fill="red")
                self.master.update()
                self.master.after(250)
                return

    def clear_canvas(self):
        # Șterge toate elementele din canvas
        self.canvas.delete("all")
        # Resetează structurile de date
        self.nodes.clear()
        self.edges.clear()
        self.graph = GraphWeighted(is_directed=self.is_directed.get())  # Reinicializăm graful
        self.node_counter = 0
        self.edge_start_node = None
        # Actualizăm meniul pentru nodul de start
        self.update_start_node_menu()

    def find_graph_centers(self):
        # Calculăm gradul fiecărui nod (numărul de vecini)
        degrees = {node_id: len(neighbors) for node_id, neighbors in self.graph.adjacency_list.items()}

        # Găsim frunzele inițiale (nodurile cu grad 1)
        leaves = deque([node_id for node_id, degree in degrees.items() if degree == 1])

        # Copiem lista de adiacență pentru a lucra pe o versiune temporară
        adjacency_list = {node_id: list(neighbors) for node_id, neighbors in self.graph.adjacency_list.items()}

        while len(adjacency_list) > 2:
            new_leaves = []
            for leaf in leaves:
                # Dacă frunza există în lista de adiacență, procesăm vecinii ei
                if leaf in adjacency_list:
                    for neighbor, _, _ in adjacency_list[leaf]:  # Ignorăm ponderile și costurile
                        adjacency_list[neighbor] = [n for n in adjacency_list[neighbor] if n[0] != leaf]
                        degrees[neighbor] -= 1
                        if degrees[neighbor] == 1:
                            new_leaves.append(neighbor)

                    # Ștergem frunza curentă
                    del adjacency_list[leaf]

                    # Evidențiem frunza în interfața grafică
                    self.canvas.itemconfig(self.nodes[leaf].circle_id, fill="lightgrey")
                    self.master.update()
                    self.master.after(500)  # Pauză pentru vizualizare

            leaves = deque(new_leaves)

        # La final, rămân unul sau două noduri, care sunt centrele grafului
        for node_id in adjacency_list.keys():
            self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="orange")  # Evidențiem centrele
        self.master.update()

        print("Centrele grafului:", list(adjacency_list.keys()))

    def eager_prims_mst(self):
        from heapq import heappush, heappop

        if not self.nodes:
            print("Nu există noduri în graf.")
            return

        start_node_id = self.start_node.get()
        if start_node_id not in self.graph.nodes:
            print("Nodul de start nu există în graf.")
            return

        visited = set()
        priority_queue = []
        mst_edges = []

        # Adăugăm muchiile inițiale
        for neighbor, weight, _ in self.graph.get_neighbors(start_node_id): # Ignorăm costul
            heappush(priority_queue, (weight, start_node_id, neighbor))

        visited.add(start_node_id)

        while priority_queue:
            weight, u, v = heappop(priority_queue)

            if v in visited:
                continue

            # Colorăm muchia MST
            self.highlight_mst_edge(u, v)
            mst_edges.append((u, v, weight))
            visited.add(v)

            # Adăugăm vecinii nodului v
            for neighbor, weight in self.graph.get_neighbors(v):
                if neighbor not in visited:
                    heappush(priority_queue, (weight, v, neighbor))

        total_cost = sum(weight for _, _, weight in mst_edges)
        print(f"Muchiile MST: {mst_edges}")
        print(f"Cost total MST: {total_cost}")

    def highlight_mst_edge(self, from_node_id, to_node_id):
        for edge in self.edges:
            if (edge.from_node_id == from_node_id and edge.to_node_id == to_node_id) or \
                    (edge.from_node_id == to_node_id and edge.to_node_id == from_node_id):
                self.canvas.itemconfig(edge.line_id, fill="green", width=3)
                self.master.update()
                self.master.after(500)  # Pauză pentru vizualizare
                return

    def kruskal_mst(self):
        if not self.nodes:
            print("Nu există noduri în graf.")
            return

        mst_edges, total_cost = self.graph.kruskal_mst()

        # Colorăm muchiile din MST
        for u, v, weight in mst_edges:
            self.highlight_mst_edge(u, v)

        print(f"Muchiile MST: {mst_edges}")
        print(f"Cost total MST: {total_cost}")

    def boruvka_mst(self):
        print("Borůvka MST: calculul începe...")

        if not self.nodes:
            print("Nu există noduri în graf.")
            return

        mst_edges, total_cost = self.graph.boruvka_mst()

        # Colorăm muchiile din MST
        for u, v, weight in mst_edges:
            self.highlight_mst_edge(u, v)

        print(f"Muchiile MST (Borůvka): {mst_edges}")
        print(f"Cost total MST (Borůvka): {total_cost}")
        
    # Add these methods to the GraphGUI class
    def add_flow_algorithms(self):
        # Add to create_widgets method after existing algorithms
        algorithms = [
            "Ford-Fulkerson (DFS)",
            "Edmonds-Karp (BFS)",
            "Min Cost Flow"
        ]
        for alg in algorithms:
            self.algorithm["menu"].add_command(
                label=alg,
                command=lambda value=alg: self.algorithm.set(value)
            )

    def run_flow_algorithm(self):
        algorithm = self.algorithm.get()
        source = self.start_node.get()

        # Ask for sink node
        sink = tk.simpledialog.askinteger("Sink Node", "Enter sink node number:")
        if sink is None or sink not in self.nodes:
            print("Invalid sink node")
            return

        # Create flow network
        flow_network = FlowNetwork(self.graph)

        if algorithm == "Ford-Fulkerson (DFS)":
            max_flow = flow_network.ford_fulkerson(source, sink, use_bfs=False)
            print(f"Maximum flow: {max_flow}")
            self.highlight_flow(flow_network.flow)

        elif algorithm == "Edmonds-Karp (BFS)":
            max_flow = flow_network.ford_fulkerson(source, sink, use_bfs=True)
            print(f"Maximum flow: {max_flow}")
            self.highlight_flow(flow_network.flow)

        elif algorithm == "Min Cost Flow":
            max_flow, min_cost = flow_network.cycle_canceling(source, sink)
            print(f"Maximum flow: {max_flow}")
            print(f"Minimum cost: {min_cost}")
            self.highlight_flow(flow_network.flow)

    def highlight_flow(self, flow):
        # Reset edge colors
        for edge in self.edges:
            self.canvas.itemconfig(edge.line_id, fill="black", width=1)

        # Highlight edges with positive flow
        for edge in self.edges:
            flow_value = flow.get((edge.from_node_id, edge.to_node_id), 0)
            cost = flow.get((edge.from_node_id, edge.to_node_id), 0)
            if flow_value > 0:
                    self.canvas.itemconfig(edge.line_id, fill="blue", width=2)
                    from_node = self.nodes[edge.from_node_id]
                    to_node = self.nodes[edge.to_node_id]
                    text_x = (from_node.x + to_node.x) / 2
                    text_y = (from_node.y + to_node.y) / 2
                    # Add text displaying flow and cost
                    self.canvas.create_text(
                        text_x, text_y + 15,
                        text=f"Flow: {flow_value}, Cost: {cost}",
                        fill="blue"
                    )


def test_graph_operations_example():
    print("=== Exemplu de test pentru operațiuni pe graf ===")

    # Creăm un graf simplu orientat
    test_graph = GraphWeighted(is_directed=True)

    # Definim muchiile grafului cu capacități și costuri
    test_graph.add_edge(0, 1, weight=2, cost=1)
    test_graph.add_edge(0, 2, weight=4, cost=1)
    test_graph.add_edge(1, 2, weight=3, cost=1)
    test_graph.add_edge(1, 3, weight=1, cost=4)
    test_graph.add_edge(2, 3, weight=6, cost=1)

    # Aplicăm algoritmul Cycle-Canceling
    flow_network = FlowNetwork(test_graph)
    max_flow, min_cost = flow_network.cycle_canceling(0, 3)

    # Afișăm rezultatele
    print("\nFluxurile finale:")
    for (u, v), f in flow_network.flow.items():
        if f > 0:
            print(f"Flux pe muchia ({u} -> {v}): {f}, Cost: {flow_network.cost[u, v]}")

    print("\nFluxurile finale (inclusiv cele cu 0):")
    for (u, v), f in flow_network.flow.items():
        print(f"Flux pe muchia ({u} -> {v}): {f}, Cost: {flow_network.cost[u, v]}")
    
    print(f"\nFlux maxim: {max_flow}, Cost minim: {min_cost}")


if __name__ == '__main__':
    #test_graph_operations_example()
    root = tk.Tk()
    app = GraphGUI(root)
    root.mainloop()