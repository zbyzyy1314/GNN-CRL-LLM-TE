"""
将 GEANT H5 数据转换为 CFR-RL 所需的格式：
  - data/GEANT        : 拓扑文件（节点、链路、权重、容量）
  - data/GEANTTM      : 训练流量矩阵文件
  - data/GEANTTM2     : 测试流量矩阵文件
"""
import numpy as np
import tables
import os

H5_PATH = "./data/tszoo/databases/GEANT/GEANT-matrix-backbone_network-15_minutes.h5"
DATA_DIR = "./data/"

# ==================== 1. 读取 H5 数据 ====================
print("[*] Reading GEANT H5 data...")
with tables.open_file(H5_PATH, 'r') as f:
    links_arr = f.root.links.read()  # shape (74,), fields: from, to
    tm = f.root.backbone_network.matrix_bandwidth_kbps.read()  # (10772, 23, 23)

num_nodes = 23
num_links = 74
num_tms = tm.shape[0]
print(f"    Nodes: {num_nodes}, Links: {num_links}, TMs: {num_tms}")

# ==================== 2. 生成拓扑文件 ====================
# 链路权重设为 1（无加权最短路径）
# 容量: GEANT 骨干网链路通常为 10 Gbps = 10,000,000 kbps
DEFAULT_CAPACITY = 10_000_000.0  # kbps
DEFAULT_WEIGHT = 1

topo_file = os.path.join(DATA_DIR, "GEANT")
print(f"\n[*] Generating topology file: {topo_file}")
with open(topo_file, 'w') as f:
    f.write(f"Node: {num_nodes}\tLink: {num_links}\n")
    f.write("link_idx\tsrc\tdst\tweight\tcapacity\n")
    for i, link in enumerate(links_arr):
        src = int(link['from']) - 1  # 1-indexed -> 0-indexed
        dst = int(link['to']) - 1
        f.write(f"{i}\t{src}\t{dst}\t{DEFAULT_WEIGHT}\t{DEFAULT_CAPACITY}\n")
print(f"    Done. {num_links} links written.")

# ==================== 3. 生成最短路径缓存 ====================
print(f"\n[*] Computing shortest paths...")
import networkx as nx
DG = nx.DiGraph()
for link in links_arr:
    src = int(link['from']) - 1
    dst = int(link['to']) - 1
    DG.add_edge(src, dst, weight=DEFAULT_WEIGHT)

shortest_paths_file = topo_file + "_shortest_paths"
with open(shortest_paths_file, 'w') as f:
    for s in range(num_nodes):
        for d in range(num_nodes):
            if s != d:
                paths = list(nx.all_shortest_paths(DG, s, d, weight='weight'))
                paths_str = str([[int(n) for n in p] for p in paths])
                f.write(f"{s}->{d}: {paths_str}\n")
print(f"    Done. Saved to {shortest_paths_file}")

# ==================== 4. 处理流量矩阵 ====================
print(f"\n[*] Processing traffic matrices...")

# 处理 NaN（对角线自流量及缺失值）→ 设为 0
tm = np.nan_to_num(tm, nan=0.0)

# 去掉全零矩阵（有些时间步可能无效）
valid_mask = ~np.all(tm == 0, axis=(1, 2))
tm = tm[valid_mask]
print(f"    Valid TMs after filtering: {tm.shape[0]} / {num_tms}")

# env.py 中的转换: self.traffic_matrices = file_values * scale * 8 / 300 / 1000
# 为了让最终值等于原始 kbps 值: file_values = original_kbps * 300 * 1000 / (scale * 8)
# 默认 scale = 100, 所以 file_values = original_kbps * 375
SCALE = 100
conversion = 300 * 1000 / (SCALE * 8)  # = 375
print(f"    Scale={SCALE}, file_value = original_kbps * {conversion}")

# ==================== 5. 划分训练/测试集并写入 ====================
split_idx = int(tm.shape[0] * 0.8)
train_tm = tm[:split_idx]
test_tm = tm[split_idx:]
print(f"    Train TMs: {train_tm.shape[0]}, Test TMs: {test_tm.shape[0]}")

def write_tm_file(filename, matrices, conversion_factor):
    """将流量矩阵写入文件，每行一个 NN 的矩阵（空格分隔）"""
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n[*] Writing {filepath} ...")
    with open(filepath, 'w') as f:
        for i in range(matrices.shape[0]):
            mat = matrices[i] * conversion_factor
            flat = mat.flatten()
            line = ' '.join([f"{v:.6f}" for v in flat])
            f.write(line + '\n')
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"    Done. {matrices.shape[0]} matrices, {file_size:.1f} MB")

write_tm_file("GEANTTM", train_tm, conversion)
write_tm_file("GEANTTM2", test_tm, conversion)

print("\n[✓] Conversion complete!")
print(f"    Topology:  {topo_file}")
print(f"    Train TM:  {DATA_DIR}GEANTTM")
print(f"    Test TM:   {DATA_DIR}GEANTTM2")
print(f"\n    Now update config.py:")
print(f"      topology_file = 'GEANT'")
print(f"      traffic_file = 'TM'")
print(f"      test_traffic_file = 'TM2'")
