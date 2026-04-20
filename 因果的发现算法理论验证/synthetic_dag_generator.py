"""
synthetic_dag_generator.py
==========================
合成 DAG 图生成器 - 用于因果发现算法的鲁棒性测试

支持多种图模型:
  1. Erdős-Rényi (ER): 纯随机图，任意两节点间以固定概率连边
  2. Scale-Free (SF): 无标度网络，含有超级节点（Hub），更符合工业系统拓扑
  3. Layered Industrial: 分层工业流程图，模拟真实选矿过程

数据生成机制:
  - 线性关系: X_j = a * X_i + noise
  - 饱和效应: X_j = x_max * (1 - exp(-k * X_i)) + noise
  - 阈值效应: X_j = 1 / (1 + exp(-slope * (X_i - threshold))) + noise
  - 倒U型关系: X_j = exp(-((X_i - optimal)^2) / (2 * width^2)) + noise
  - 多项式非线性: X_j = a * X_i^2 + b * X_i + noise
  - 交互作用: X_j = alpha * X_i1 * X_i2 + noise

因果角色识别:
  - 混杂变量 (Confounder): C → X, C → Y
  - 中介变量 (Mediator): X → M → Y
  - 工具变量 (Instrumental Variable): Z → X, Z ⊥ Y | X
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List, Set, Optional
import warnings
warnings.filterwarnings("ignore")


class SyntheticDAGGenerator:
    """合成 DAG 生成器"""
    
    def __init__(self, n_nodes: int = 20, seed: int = 42):
        """
        参数:
            n_nodes: 节点数量（变量数）
            seed: 随机种子，保证可重复性
        """
        self.n_nodes = n_nodes
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.causal_roles = {}  # 存储因果角色信息
        
    def generate_er_dag(self, edge_prob: float = 0.2) -> np.ndarray:
        """
        生成 Erdős-Rényi 随机 DAG
        
        参数:
            edge_prob: 任意两节点间连边概率
            
        返回:
            adj_matrix: (n, n) 邻接矩阵，adj[i,j]=1 表示 i->j 有边
        """
        # 生成随机图
        G = nx.erdos_renyi_graph(self.n_nodes, edge_prob, seed=self.seed, directed=True)
        
        # 转换为 DAG（移除环）
        adj = nx.to_numpy_array(G)
        
        # 强制转为 DAG：只保留 i < j 的边（上三角矩阵）
        adj = np.triu(adj, k=1)
        
        return adj
    
    def generate_scale_free_dag(self, m: int = 2) -> np.ndarray:
        """
        生成 Scale-Free 无标度 DAG（含超级节点）
        改进：不再随机分配方向，而是基于节点度数决定方向
        
        参数:
            m: 每个新节点连接到已有节点的边数
            
        返回:
            adj_matrix: (n, n) 邻接矩阵
        """
        # 使用 Barabási-Albert 模型生成无标度网络
        G = nx.barabasi_albert_graph(self.n_nodes, m, seed=self.seed)
        
        # 转为有向图
        G_directed = nx.DiGraph()
        G_directed.add_nodes_from(G.nodes())
        
        # 计算每个节点的度数（用于决定边方向）
        degrees = dict(G.degree())
        
        # 为每条无向边分配方向：从低度节点指向高度节点（模拟因果流）
        for u, v in G.edges():
            if degrees[u] < degrees[v]:
                G_directed.add_edge(u, v)
            elif degrees[u] > degrees[v]:
                G_directed.add_edge(v, u)
            else:
                # 度数相同时，从节点序号小的指向大的
                if u < v:
                    G_directed.add_edge(u, v)
                else:
                    G_directed.add_edge(v, u)
        
        # 转为邻接矩阵并确保 DAG（移除可能的环）
        adj = nx.to_numpy_array(G_directed)
        adj = self._ensure_dag(adj)
        adj = self._ensure_connectivity(adj)
        
        return adj
    
    def _ensure_dag(self, adj: np.ndarray) -> np.ndarray:
        """
        确保邻接矩阵是 DAG（无环）
        改进：使用拓扑排序而非简单的上三角化
        """
        G = nx.DiGraph(adj)
        
        # 检查是否有环
        if nx.is_directed_acyclic_graph(G):
            return adj
        
        # 如果有环，使用拓扑排序重新排列节点
        try:
            # 移除环：找到所有强连通分量
            sccs = list(nx.strongly_connected_components(G))
            
            # 对于每个强连通分量（环），只保留部分边
            for scc in sccs:
                if len(scc) > 1:
                    scc_list = list(scc)
                    # 在环内只保留单向链
                    for i in range(len(scc_list)):
                        for j in range(i+1, len(scc_list)):
                            # 移除反向边
                            if G.has_edge(scc_list[j], scc_list[i]):
                                G.remove_edge(scc_list[j], scc_list[i])
            
            return nx.to_numpy_array(G)
        except:
            # 如果失败，回退到上三角化
            return np.triu(adj, k=1)
    
    def _ensure_connectivity(self, adj: np.ndarray) -> np.ndarray:
        """
        确保每个节点至少有一个父节点或子节点（除了根节点和叶节点）
        """
        for node in range(self.n_nodes):
            in_degree = adj[:, node].sum()
            out_degree = adj[node, :].sum()
            
            if in_degree == 0 and out_degree == 0:
                # 孤立节点：随机连接
                if node > 0:
                    # 连接到一个前驱节点
                    parent = self.rng.choice(node)
                    adj[parent, node] = 1
                elif node < self.n_nodes - 1:
                    # 连接到一个后继节点
                    child = self.rng.choice(range(node+1, self.n_nodes))
                    adj[node, child] = 1
        
        return adj
    
    def generate_layered_industrial_dag(
        self,
        n_layers: int = 5,
        nodes_per_layer: Optional[List[int]] = None,
        inter_layer_prob: float = 0.3,
        intra_layer_prob: float = 0.1
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        生成分层工业 DAG，模拟真实选矿流程
        
        层次结构示例：
        Layer 0: 原料特性（矿石品位、粒度、硬度）
        Layer 1: 破碎/研磨参数（给矿量、转速、钢球配比）
        Layer 2: 选别参数（药剂添加量、浮选时间、pH值）
        Layer 3: 中间指标（泡沫层厚度、矿浆浓度）
        Layer 4: 最终产品质量（精矿品位、回收率）
        
        参数:
            n_layers: 工艺层数
            nodes_per_layer: 每层节点数列表，如果为None则自动分配
            inter_layer_prob: 跨层连接概率
            intra_layer_prob: 同层连接概率（横向影响）
            
        返回:
            adj_matrix: 邻接矩阵
            layer_indices: 每层的节点索引列表
        """
        if nodes_per_layer is None:
            if self.n_nodes < n_layers:
                raise ValueError(
                    f"n_nodes ({self.n_nodes}) 必须 >= n_layers ({n_layers})"
                )
            # 自动分配：中间层节点多，首尾层节点少
            base = self.n_nodes // n_layers
            nodes_per_layer = [base] * n_layers
            remainder = self.n_nodes - base * n_layers
            # 将余数从中间向两侧分配，防止索引越界
            mid = n_layers // 2
            for i in range(remainder):
                # 交替在 mid, mid+1, mid-1, mid+2, ... 位置分配
                if i % 2 == 0:
                    idx = mid + i // 2
                else:
                    idx = mid - (i // 2 + 1)
                idx = max(0, min(idx, n_layers - 1))
                nodes_per_layer[idx] += 1
        
        total_nodes = sum(nodes_per_layer)
        if total_nodes != self.n_nodes:
            raise ValueError(f"nodes_per_layer 总和 ({total_nodes}) 必须等于 n_nodes ({self.n_nodes})")
        
        adj = np.zeros((self.n_nodes, self.n_nodes))
        
        # 构建层索引
        layer_indices = []
        start = 0
        for n in nodes_per_layer:
            layer_indices.append(list(range(start, start + n)))
            start += n
        
        # 跨层连接（前向因果流）
        for l in range(n_layers - 1):
            for i in layer_indices[l]:
                # 每个节点至少连接到下一层的一个节点
                n_connections = max(1, self.rng.poisson(2))
                targets = self.rng.choice(
                    layer_indices[l + 1],
                    size=min(n_connections, len(layer_indices[l + 1])),
                    replace=False
                )
                for j in targets:
                    adj[i, j] = 1
                
                # 额外的随机连接
                for j in layer_indices[l + 1]:
                    if adj[i, j] == 0 and self.rng.rand() < inter_layer_prob:
                        adj[i, j] = 1
        
        # 同层连接（有序，模拟并行工艺的相互影响）
        for l in range(n_layers):
            nodes = layer_indices[l]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.rng.rand() < intra_layer_prob:
                        adj[nodes[i], nodes[j]] = 1
        
        return adj, layer_indices
    
    def assign_edge_functions(
        self, 
        adj: np.ndarray,
        layer_indices: Optional[List[List[int]]] = None,
        use_industrial_functions: bool = False
    ) -> Dict[Tuple[int, int], Dict]:
        """
        为 DAG 中的每条边分配因果函数类型和参数
        
        参数:
            adj: 邻接矩阵
            layer_indices: 层索引（如果提供，将根据层关系选择函数类型）
            use_industrial_functions: 是否使用工业相关的函数类型
            
        返回:
            edge_funcs: {(i, j): {'type': 'linear'/'saturation'/..., 'params': {...}}}
        """
        edge_funcs = {}
        
        # 确定节点所在层（如果有）
        node_to_layer = {}
        if layer_indices is not None:
            for layer_idx, nodes in enumerate(layer_indices):
                for node in nodes:
                    node_to_layer[node] = layer_idx
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if adj[i, j] > 0:  # 存在边 i -> j
                    
                    if use_industrial_functions:
                        func_type, params = self._assign_industrial_function(
                            i, j, node_to_layer
                        )
                    else:
                        func_type, params = self._assign_standard_function()
                    
                    edge_funcs[(i, j)] = {'type': func_type, 'params': params}
        
        return edge_funcs
    
    def _assign_standard_function(self) -> Tuple[str, Dict]:
        """分配标准函数类型（原有方法）"""
        func_type = self.rng.choice(
            ['linear', 'sin', 'exp', 'poly'],
            p=[0.30, 0.30, 0.20, 0.20]
        )
        
        if func_type == 'linear':
            params = {'a': self.rng.uniform(0.5, 2.0) * self.rng.choice([-1, 1])}
        elif func_type == 'sin':
            params = {'b': self.rng.uniform(0.5, 2.0)}
        elif func_type == 'exp':
            params = {'c': self.rng.uniform(0.1, 0.5)}
        elif func_type == 'poly':
            params = {
                'a': self.rng.uniform(0.1, 0.5) * self.rng.choice([-1, 1]),
                'b': self.rng.uniform(0.5, 1.5) * self.rng.choice([-1, 1])
            }
        
        return func_type, params
    
    def _assign_industrial_function(
        self, 
        i: int, 
        j: int, 
        node_to_layer: Dict[int, int]
    ) -> Tuple[str, Dict]:
        """
        分配工业相关的函数类型
        根据节点所在层和边的类型选择合适的非线性关系
        """
        i_layer = node_to_layer.get(i, 0)
        j_layer = node_to_layer.get(j, 0)
        n_layers = max(node_to_layer.values()) + 1 if node_to_layer else 1
        
        # 最终产品质量层：更可能是饱和/阈值效应
        if j_layer == n_layers - 1:
            func_type = self.rng.choice(
                ['saturation', 'threshold', 'inverted_u', 'linear'],
                p=[0.35, 0.25, 0.25, 0.15]
            )
        # 同层连接：线性或弱非线性
        elif i_layer == j_layer:
            func_type = self.rng.choice(
                ['linear', 'poly'],
                p=[0.7, 0.3]
            )
        # 跨层连接：混合
        else:
            func_type = self.rng.choice(
                ['linear', 'saturation', 'threshold', 'poly', 'inverted_u'],
                p=[0.30, 0.25, 0.15, 0.20, 0.10]
            )
        
        # 生成参数
        if func_type == 'linear':
            params = {'a': self.rng.uniform(0.5, 2.0) * self.rng.choice([-1, 1])}
        elif func_type == 'saturation':
            params = {
                'k': self.rng.uniform(1.0, 3.0),
                'x_max': self.rng.uniform(3.0, 6.0)
            }
        elif func_type == 'threshold':
            params = {
                'threshold': self.rng.uniform(-1.0, 1.0),
                'slope': self.rng.uniform(2.0, 5.0)
            }
        elif func_type == 'inverted_u':
            params = {
                'optimal': self.rng.uniform(-1.0, 1.0),
                'width': self.rng.uniform(1.0, 3.0)
            }
        elif func_type == 'poly':
            params = {
                'a': self.rng.uniform(0.1, 0.5) * self.rng.choice([-1, 1]),
                'b': self.rng.uniform(0.5, 1.5) * self.rng.choice([-1, 1])
            }
        
        return func_type, params
    
    def generate_data(
        self, 
        adj: np.ndarray, 
        edge_funcs: Dict[Tuple[int, int], Dict],
        n_samples: int = 1000,
        noise_scale: float = 0.1,
        noise_type: str = 'gaussian',
        add_time_lag: bool = False,
        lag_order: int = 1,
        do_interventions: Optional[Dict[int, np.ndarray]] = None
    ) -> np.ndarray:
        """
        根据 DAG 结构和边函数生成时序数据
        
        参数:
            adj: 邻接矩阵
            edge_funcs: 边函数字典
            n_samples: 样本数（时间步数）
            noise_scale: 噪声标准差
            noise_type: 噪声类型 ('gaussian', 'heteroscedastic', 'heavy_tail', 'periodic')
            add_time_lag: 是否添加时序依赖（自回归）
            lag_order: 自回归阶数
            do_interventions: do-calculus 干预字典，格式 {node_idx: np.ndarray(n_samples,)}
                当提供时，指定节点的值被强制设为给定值（忽略其父节点的因果作用），
                但 RNG 调用序列保持不变，确保与未干预生成的噪声完全同步。
                用于仿真 do(T=t) 计算真实因果效应。
            
        返回:
            X: (n_samples, n_nodes) 数据矩阵
        """
        X = np.zeros((n_samples, self.n_nodes))
        
        # 拓扑排序，确保按因果顺序生成数据
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_nodes))  # 确保所有节点都被包含
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if adj[i, j] > 0:
                    G.add_edge(i, j)
        
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            topo_order = list(range(self.n_nodes))
        
        # 按时间步生成数据
        for t in range(n_samples):
            for node in topo_order:
                # 找到所有父节点
                parents = [i for i in range(self.n_nodes) if adj[i, node] > 0]
                
                # 自回归项（时序依赖）
                ar_term = 0.0
                if add_time_lag and t >= lag_order:
                    for lag in range(1, lag_order + 1):
                        # 衰减系数：越远的滞后影响越小
                        decay = 0.3 ** lag
                        ar_term += decay * X[t - lag, node]
                
                # 因果项（来自父节点）
                if len(parents) == 0:
                    # 根节点：从标准正态分布采样
                    causal_term = self.rng.randn()
                else:
                    # 非根节点：根据父节点和边函数计算
                    causal_term = self._compute_causal_contribution(
                        X, t, node, parents, edge_funcs
                    )
                
                # 生成噪声（传入当前信号强度，用于异方差噪声）
                current_signal = ar_term + causal_term
                noise = self._generate_noise(
                    t, node, X, noise_scale, noise_type,
                    current_signal=current_signal
                )
                
                # do-calculus 干预：强制设置节点值，忽略因果计算结果
                # 注意：上面的 RNG 调用（randn / _generate_noise）仍然执行，
                # 保证 RNG 序列与非干预生成完全同步
                if do_interventions is not None and node in do_interventions:
                    X[t, node] = do_interventions[node][t]
                else:
                    # 组合并裁剪，防止数值爆炸
                    X[t, node] = np.clip(
                        ar_term + causal_term + noise, -1e4, 1e4
                    )
        
        # 最终安全检查：替换可能残留的 NaN/Inf
        if np.any(~np.isfinite(X)):
            warnings.warn("生成的数据中存在 NaN/Inf，已替换为 0")
            X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return X
    
    def _compute_causal_contribution(
        self,
        X: np.ndarray,
        t: int,
        node: int,
        parents: List[int],
        edge_funcs: Dict[Tuple[int, int], Dict]
    ) -> float:
        """计算父节点对当前节点的因果贡献（按父节点数归一化防止数值爆炸）"""
        contribution = 0.0
        
        for parent in parents:
            func_info = edge_funcs[(parent, node)]
            func_type = func_info['type']
            params = func_info['params']
            
            parent_data = X[t, parent]
            
            if func_type == 'linear':
                contribution += params['a'] * parent_data
            elif func_type == 'sin':
                contribution += np.sin(params['b'] * parent_data)
            elif func_type == 'exp':
                contribution += np.exp(-params['c'] * np.abs(parent_data))
            elif func_type == 'saturation':
                contribution += params['x_max'] * (1 - np.exp(-params['k'] * np.abs(parent_data)))
            elif func_type == 'threshold':
                contribution += 1 / (1 + np.exp(-params['slope'] * (parent_data - params['threshold'])))
            elif func_type == 'inverted_u':
                contribution += np.exp(-((parent_data - params['optimal']) ** 2) / (2 * params['width'] ** 2))
            elif func_type == 'poly':
                contribution += params['a'] * parent_data**2 + params['b'] * parent_data
        
        # 按父节点数归一化，防止多父节点累加导致数值逐层爆炸
        if len(parents) > 1:
            contribution /= len(parents)
        
        return contribution
    
    def _generate_noise(
        self,
        t: int,
        node: int,
        X: np.ndarray,
        noise_scale: float,
        noise_type: str,
        current_signal: float = 0.0
    ) -> float:
        """
        生成不同类型的噪声
        
        参数:
            t: 当前时间步
            node: 当前节点
            X: 数据矩阵
            noise_scale: 噪声尺度
            noise_type: 噪声类型
            current_signal: 当前节点的信号值（用于异方差噪声）
        """
        if noise_type == 'gaussian':
            # 标准高斯白噪声
            return self.rng.randn() * noise_scale
        
        elif noise_type == 'heteroscedastic':
            # 异方差噪声：噪声大小依赖于当前信号强度
            base_noise = self.rng.randn()
            scale = noise_scale * (1.0 + 0.5 * np.abs(current_signal))
            return base_noise * scale
        
        elif noise_type == 'heavy_tail':
            # 重尾噪声（t分布）：模拟异常工况
            return self.rng.standard_t(df=3) * noise_scale
        
        elif noise_type == 'periodic':
            # 周期性噪声：模拟班次/季节影响
            base_noise = self.rng.randn() * noise_scale
            period = 24  # 假设24小时周期
            periodic_component = 0.2 * noise_scale * np.sin(2 * np.pi * t / period)
            return base_noise + periodic_component
        
        else:
            return self.rng.randn() * noise_scale
    
    def generate_complete_synthetic_dataset(
        self,
        graph_type: str = 'scale_free',
        edge_prob: float = 0.2,
        sf_m: int = 2,
        n_samples: int = 1000,
        noise_scale: float = 0.1,
        noise_type: str = 'gaussian',
        add_time_lag: bool = False,
        use_industrial_functions: bool = False,
        n_layers: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        一站式生成完整的合成数据集
        
        参数:
            graph_type: 'er', 'scale_free', 或 'layered'
            edge_prob: ER 图的连边概率
            sf_m: Scale-Free 图的参数
            n_samples: 样本数
            noise_scale: 噪声水平
            noise_type: 噪声类型
            add_time_lag: 是否添加时序依赖
            use_industrial_functions: 是否使用工业函数
            n_layers: 分层图的层数
            
        返回:
            X: 数据矩阵
            adj_true: 真实邻接矩阵
            metadata: 元数据（边函数、层信息等）
        """
        # 生成 DAG 结构
        layer_indices = None
        if graph_type == 'er':
            adj_true = self.generate_er_dag(edge_prob)
        elif graph_type == 'scale_free':
            adj_true = self.generate_scale_free_dag(sf_m)
        elif graph_type == 'layered':
            adj_true, layer_indices = self.generate_layered_industrial_dag(n_layers)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        # 分配边函数
        edge_funcs = self.assign_edge_functions(
            adj_true, 
            layer_indices, 
            use_industrial_functions
        )
        
        # 生成数据
        X = self.generate_data(
            adj_true, 
            edge_funcs, 
            n_samples, 
            noise_scale,
            noise_type,
            add_time_lag
        )
        
        # 元数据
        metadata = {
            'n_nodes': self.n_nodes,
            'n_edges': int(adj_true.sum()),
            'graph_type': graph_type,
            'edge_funcs': edge_funcs,
            'layer_indices': layer_indices,
            'seed': self.seed,
            'noise_type': noise_type,
            'add_time_lag': add_time_lag
        }
        
        return X, adj_true, metadata
    
    def identify_causal_roles(
        self,
        adj: np.ndarray,
        treatment_idx: int,
        outcome_idx: int
    ) -> Dict[str, List[int]]:
        """
        识别给定 treatment-outcome 对的因果角色
        
        参数:
            adj: 邻接矩阵
            treatment_idx: 处理变量（操作变量）索引
            outcome_idx: 结果变量索引
            
        返回:
            roles: {
                'confounders': 混杂变量列表,
                'mediators': 中介变量列表,
                'instruments': 工具变量列表,
                'colliders': 对撞变量列表
            }
        """
        G = nx.DiGraph(adj)
        
        confounders = []
        mediators = []
        instruments = []
        colliders = []
        
        for node in range(self.n_nodes):
            if node == treatment_idx or node == outcome_idx:
                continue
            
            # 1. 混杂变量：同时指向 X 和 Y
            if (G.has_edge(node, treatment_idx) and 
                G.has_edge(node, outcome_idx)):
                confounders.append(node)
            
            # 2. 中介变量：在 X → Y 的路径上
            elif (G.has_edge(treatment_idx, node) and 
                  nx.has_path(G, node, outcome_idx)):
                mediators.append(node)
            
            # 3. 工具变量候选：Z → X 但 Z 不直接或间接影响 Y（除了通过 X）
            elif G.has_edge(node, treatment_idx):
                # 检查是否有不经过 X 的路径到 Y
                G_without_X = G.copy()
                G_without_X.remove_node(treatment_idx)
                if not nx.has_path(G_without_X, node, outcome_idx):
                    instruments.append(node)
            
            # 4. 对撞变量：X → C ← Y
            if (G.has_edge(treatment_idx, node) and 
                G.has_edge(outcome_idx, node)):
                colliders.append(node)
        
        return {
            'confounders': confounders,
            'mediators': mediators,
            'instruments': instruments,
            'colliders': colliders
        }
    
    def find_adjustment_set(
        self,
        adj: np.ndarray,
        treatment_idx: int,
        outcome_idx: int
    ) -> Set[int]:
        """
        找到最小调整集（用于去混杂）
        
        使用后门准则：
        1. 阻断所有后门路径（X ← ... → Y）
        2. 不打开任何对撞路径
        """
        roles = self.identify_causal_roles(adj, treatment_idx, outcome_idx)
        
        # 简化版本：返回所有混杂变量
        # 更复杂的实现需要考虑对撞变量
        adjustment_set = set(roles['confounders'])
        
        return adjustment_set


def compute_dag_metrics(adj_true: np.ndarray, adj_pred: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    """
    计算 DAG 恢复的评估指标
    
    参数:
        adj_true: 真实邻接矩阵 (n, n)
        adj_pred: 预测邻接矩阵 (n, n)，可以是连续权重
        threshold: 预测矩阵的二值化阈值
        
    返回:
        metrics: {'SHD': ..., 'TPR': ..., 'FDR': ..., 'Precision': ..., 'Recall': ...}
    """
    # 二值化
    adj_true_bin = (adj_true > 0).astype(int)
    adj_pred_bin = (np.abs(adj_pred) > threshold).astype(int)
    
    # 真阳性、假阳性、假阴性
    TP = np.sum((adj_true_bin == 1) & (adj_pred_bin == 1))
    FP = np.sum((adj_true_bin == 0) & (adj_pred_bin == 1))
    FN = np.sum((adj_true_bin == 1) & (adj_pred_bin == 0))
    TN = np.sum((adj_true_bin == 0) & (adj_pred_bin == 0))
    
    # SHD (Structural Hamming Distance)
    SHD = FP + FN
    
    # TPR (True Positive Rate / Recall / Sensitivity)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # FDR (False Discovery Rate)
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Precision
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # F1 Score
    F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0.0
    
    return {
        'SHD': float(SHD),
        'TPR': float(TPR),
        'FDR': float(FDR),
        'Precision': float(Precision),
        'Recall': float(TPR),
        'F1': float(F1),
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN)
    }


def evaluate_causal_structure_recovery(
    adj_true: np.ndarray,
    adj_pred: np.ndarray,
    treatment_idx: int,
    outcome_idx: int,
    threshold: float = 0.05
) -> Dict[str, float]:
    """
    评估因果结构恢复的质量（面向操作变量）
    
    不仅看边的准确性，更看因果角色的识别准确性
    
    参数:
        adj_true: 真实邻接矩阵
        adj_pred: 预测邻接矩阵
        treatment_idx: 处理变量（操作变量）索引
        outcome_idx: 结果变量索引
        threshold: 二值化阈值
        
    返回:
        metrics: 包含因果角色识别准确率的指标字典
    """
    # 二值化预测矩阵
    adj_pred_bin = (np.abs(adj_pred) > threshold).astype(float)
    
    # 创建生成器实例用于角色识别
    gen = SyntheticDAGGenerator(n_nodes=adj_true.shape[0])
    
    # 识别真实的因果角色
    true_roles = gen.identify_causal_roles(adj_true, treatment_idx, outcome_idx)
    
    # 识别预测的因果角色
    pred_roles = gen.identify_causal_roles(adj_pred_bin, treatment_idx, outcome_idx)
    
    metrics = {}
    
    # 辅助函数：计算集合的 Precision 和 Recall
    def compute_set_metrics(true_set: List[int], pred_set: List[int]) -> Tuple[float, float, float]:
        true_set = set(true_set)
        pred_set = set(pred_set)
        
        if len(pred_set) == 0:
            precision = 1.0 if len(true_set) == 0 else 0.0
        else:
            precision = len(true_set & pred_set) / len(pred_set)
        
        if len(true_set) == 0:
            recall = 1.0 if len(pred_set) == 0 else 0.0
        else:
            recall = len(true_set & pred_set) / len(true_set)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return precision, recall, f1
    
    # 1. 混杂变量识别准确率
    conf_prec, conf_rec, conf_f1 = compute_set_metrics(
        true_roles['confounders'], 
        pred_roles['confounders']
    )
    metrics['confounder_precision'] = conf_prec
    metrics['confounder_recall'] = conf_rec
    metrics['confounder_f1'] = conf_f1
    metrics['n_true_confounders'] = len(true_roles['confounders'])
    metrics['n_pred_confounders'] = len(pred_roles['confounders'])
    
    # 2. 中介变量识别准确率
    med_prec, med_rec, med_f1 = compute_set_metrics(
        true_roles['mediators'], 
        pred_roles['mediators']
    )
    metrics['mediator_precision'] = med_prec
    metrics['mediator_recall'] = med_rec
    metrics['mediator_f1'] = med_f1
    metrics['n_true_mediators'] = len(true_roles['mediators'])
    metrics['n_pred_mediators'] = len(pred_roles['mediators'])
    
    # 3. 工具变量识别准确率
    inst_prec, inst_rec, inst_f1 = compute_set_metrics(
        true_roles['instruments'], 
        pred_roles['instruments']
    )
    metrics['instrument_precision'] = inst_prec
    metrics['instrument_recall'] = inst_rec
    metrics['instrument_f1'] = inst_f1
    metrics['n_true_instruments'] = len(true_roles['instruments'])
    metrics['n_pred_instruments'] = len(pred_roles['instruments'])
    
    # 4. 直接效应识别
    true_direct = adj_true[treatment_idx, outcome_idx] > 0
    pred_direct = adj_pred_bin[treatment_idx, outcome_idx] > 0
    metrics['direct_effect_correct'] = float(true_direct == pred_direct)
    
    # 5. 调整集识别
    true_adj_set = gen.find_adjustment_set(adj_true, treatment_idx, outcome_idx)
    pred_adj_set = gen.find_adjustment_set(adj_pred_bin, treatment_idx, outcome_idx)
    
    if len(true_adj_set) == 0 and len(pred_adj_set) == 0:
        metrics['adjustment_set_correct'] = 1.0
    elif len(true_adj_set) == 0 or len(pred_adj_set) == 0:
        metrics['adjustment_set_correct'] = 0.0
    else:
        metrics['adjustment_set_correct'] = float(true_adj_set == pred_adj_set)
    
    # 6. 综合因果结构得分
    metrics['causal_structure_score'] = (
        0.4 * conf_f1 + 
        0.3 * med_f1 + 
        0.2 * inst_f1 + 
        0.1 * metrics['direct_effect_correct']
    )
    
    return metrics


def evaluate_multiple_treatments(
    adj_true: np.ndarray,
    adj_pred: np.ndarray,
    treatment_indices: List[int],
    outcome_idx: int,
    threshold: float = 0.05
) -> Dict[str, any]:
    """
    评估多个操作变量的因果结构恢复
    
    参数:
        adj_true: 真实邻接矩阵
        adj_pred: 预测邻接矩阵
        treatment_indices: 多个处理变量索引列表
        outcome_idx: 结果变量索引
        threshold: 二值化阈值
        
    返回:
        results: 包含每个 treatment 的评估结果和汇总统计
    """
    results = {
        'per_treatment': [],
        'summary': {}
    }
    
    # 对每个操作变量分别评估
    for treatment_idx in treatment_indices:
        metrics = evaluate_causal_structure_recovery(
            adj_true, adj_pred, treatment_idx, outcome_idx, threshold
        )
        metrics['treatment_idx'] = treatment_idx
        results['per_treatment'].append(metrics)
    
    # 汇总统计
    if len(results['per_treatment']) > 0:
        import pandas as pd
        df = pd.DataFrame(results['per_treatment'])
        
        results['summary'] = {
            'avg_confounder_f1': df['confounder_f1'].mean(),
            'avg_mediator_f1': df['mediator_f1'].mean(),
            'avg_instrument_f1': df['instrument_f1'].mean(),
            'avg_causal_structure_score': df['causal_structure_score'].mean(),
            'direct_effect_accuracy': df['direct_effect_correct'].mean()
        }
    
    return results


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("合成 DAG 生成器测试 - 改进版")
    print("=" * 70)
    
    # ========== 测试 1: 分层工业 DAG ==========
    print("\n" + "=" * 70)
    print("测试 1: 分层工业 DAG（模拟选矿流程）")
    print("=" * 70)
    
    gen = SyntheticDAGGenerator(n_nodes=20, seed=42)
    X, adj_true, metadata = gen.generate_complete_synthetic_dataset(
        graph_type='layered',
        n_samples=1000,
        noise_scale=0.1,
        noise_type='heteroscedastic',
        add_time_lag=True,
        use_industrial_functions=True,
        n_layers=5
    )
    
    print(f"\n生成的数据集:")
    print(f"  节点数: {metadata['n_nodes']}")
    print(f"  边数: {metadata['n_edges']}")
    print(f"  样本数: {X.shape[0]}")
    print(f"  图类型: {metadata['graph_type']}")
    print(f"  噪声类型: {metadata['noise_type']}")
    print(f"  时序依赖: {metadata['add_time_lag']}")
    
    if metadata['layer_indices']:
        print(f"\n层结构:")
        for i, layer in enumerate(metadata['layer_indices']):
            print(f"  Layer {i}: {len(layer)} 个节点 {layer}")
    
    print(f"\n边函数分布:")
    func_types = [info['type'] for info in metadata['edge_funcs'].values()]
    unique_types = set(func_types)
    for ftype in unique_types:
        count = func_types.count(ftype)
        print(f"  {ftype}: {count} ({count/len(func_types)*100:.1f}%)")
    
    print(f"\n数据统计:")
    print(f"  均值: {X.mean(axis=0).mean():.4f}")
    print(f"  标准差: {X.std(axis=0).mean():.4f}")
    print(f"  最小值: {X.min():.4f}")
    print(f"  最大值: {X.max():.4f}")
    
    # ========== 测试 2: 因果角色识别 ==========
    print("\n" + "=" * 70)
    print("测试 2: 因果角色识别")
    print("=" * 70)
    
    # 选择操作变量和结果变量
    treatment_idx = 5  # 假设是某个工艺参数
    outcome_idx = 19   # 假设是最终产品质量
    
    roles = gen.identify_causal_roles(adj_true, treatment_idx, outcome_idx)
    
    print(f"\n对于 Treatment={treatment_idx}, Outcome={outcome_idx}:")
    print(f"  混杂变量 (Confounders): {roles['confounders']} (共 {len(roles['confounders'])} 个)")
    print(f"  中介变量 (Mediators): {roles['mediators']} (共 {len(roles['mediators'])} 个)")
    print(f"  工具变量 (Instruments): {roles['instruments']} (共 {len(roles['instruments'])} 个)")
    print(f"  对撞变量 (Colliders): {roles['colliders']} (共 {len(roles['colliders'])} 个)")
    
    adjustment_set = gen.find_adjustment_set(adj_true, treatment_idx, outcome_idx)
    print(f"  最小调整集: {adjustment_set}")
    
    # ========== 测试 3: 因果结构评估 ==========
    print("\n" + "=" * 70)
    print("测试 3: 因果结构恢复评估")
    print("=" * 70)
    
    # 模拟一个预测的 DAG（添加噪声）
    adj_pred = adj_true.copy()
    # 添加一些错误
    noise_mask = gen.rng.rand(*adj_true.shape) < 0.1
    adj_pred[noise_mask] = 1 - adj_pred[noise_mask]
    adj_pred = np.triu(adj_pred, k=1)  # 确保是 DAG
    
    # 传统指标
    traditional_metrics = compute_dag_metrics(adj_true, adj_pred, threshold=0.5)
    print(f"\n传统 DAG 恢复指标:")
    print(f"  SHD: {traditional_metrics['SHD']}")
    print(f"  Precision: {traditional_metrics['Precision']:.3f}")
    print(f"  Recall: {traditional_metrics['Recall']:.3f}")
    print(f"  F1: {traditional_metrics['F1']:.3f}")
    
    # 因果结构指标
    causal_metrics = evaluate_causal_structure_recovery(
        adj_true, adj_pred, treatment_idx, outcome_idx, threshold=0.5
    )
    print(f"\n因果结构恢复指标 (Treatment={treatment_idx}, Outcome={outcome_idx}):")
    print(f"  混杂变量 F1: {causal_metrics['confounder_f1']:.3f} "
          f"(真实: {causal_metrics['n_true_confounders']}, "
          f"预测: {causal_metrics['n_pred_confounders']})")
    print(f"  中介变量 F1: {causal_metrics['mediator_f1']:.3f} "
          f"(真实: {causal_metrics['n_true_mediators']}, "
          f"预测: {causal_metrics['n_pred_mediators']})")
    print(f"  工具变量 F1: {causal_metrics['instrument_f1']:.3f} "
          f"(真实: {causal_metrics['n_true_instruments']}, "
          f"预测: {causal_metrics['n_pred_instruments']})")
    print(f"  直接效应正确: {causal_metrics['direct_effect_correct']:.0f}")
    print(f"  调整集正确: {causal_metrics['adjustment_set_correct']:.0f}")
    print(f"  综合因果结构得分: {causal_metrics['causal_structure_score']:.3f}")
    
    # ========== 测试 4: 多操作变量评估 ==========
    print("\n" + "=" * 70)
    print("测试 4: 多操作变量评估")
    print("=" * 70)
    
    treatment_indices = [3, 5, 8, 12]  # 多个工艺参数
    multi_results = evaluate_multiple_treatments(
        adj_true, adj_pred, treatment_indices, outcome_idx, threshold=0.5
    )
    
    print(f"\n对 {len(treatment_indices)} 个操作变量的平均性能:")
    for key, value in multi_results['summary'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
