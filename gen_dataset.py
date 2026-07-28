"""
gen_dataset.py — 200 节点数据集生成器(拓扑 + 流量,不分两个文件)

生成的内容(放在 --out-dir 下):
    large200_topo           拓扑文件,te_framework.Traology 格式
    large200_train.txt      训练流量,te_framework.TrafficLoader 格式
    large200_test.txt       测试流量

Usage:
    # 默认 200 节点 + Waxman + Gravity-lognormal
    python gen_dataset.py --nodes 200

    # 100 节点 + BA 无标度
    python gen_dataset.py --nodes 100 --model ba --ba-m 3

    # 自定义路径与规模
    python gen_dataset.py --nodes 200 --out-dir my_data \\
        --num-train 1000 --num-test 200 --total-kbps 1e8

接入 train_direct(只改路径):
    # 原来
    topo = Topology('data/LMTE_correct_topo', max_paths_per_pair=8)
    train = TrafficLoader('data/LMTE_train.txt', topo.num_nodes)
    test = TrafficLoader('data/LMTE_test.txt', topo.num_nodes)

    # 改成(注意:文件名带节点数)
    topo = Topology('data/large200_topo', max_paths_per_pair=8)
    train = TrafficLoader('data/large200_train.txt', topo.num_nodes)
    test = TrafficLoader('data/large200_test.txt', topo.num_nodes)
"""

import os
import argparse
import numpy as np
import networkx as nx


# ============================================================
#  拓扑生成
# ============================================================
def build_waxman(n, alpha=0.3, beta=0.4, seed=42, base_cap=1000.0):
    g_und = nx.waxman_graph(n, beta=beta, alpha=alpha, seed=seed)
    pos = nx.get_node_attributes(g_und, 'pos')
    if not pos:
        rng = np.random.default_rng(seed)
        pos = {i: (rng.random(), rng.random()) for i in range(n)}
    dg = nx.DiGraph(); dg.add_nodes_from(range(n))
    for u, v in g_und.edges():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        w = max(1, int(round(np.hypot(x2 - x1, y2 - y1) * 1000)))
        deg = g_und.degree(u) + g_und.degree(v)
        cap = base_cap * (1.0 + 0.2 * deg)
        dg.add_edge(u, v, weight=w, capacity=cap)
        dg.add_edge(v, u, weight=w, capacity=cap)
    return dg


def build_ba(n, m=3, seed=42, base_cap=1000.0):
    g_und = nx.barabasi_albert_graph(n, m=m, seed=seed)
    dg = nx.DiGraph(); dg.add_nodes_from(range(n))
    for u, v in g_und.edges():
        deg = g_und.degree(u) + g_und.degree(v)
        cap = base_cap * (1.0 + 0.3 * deg)
        dg.add_edge(u, v, weight=1, capacity=cap)
        dg.add_edge(v, u, weight=1, capacity=cap)
    return dg


def build_grid(n, seed=42, base_cap=1000.0):
    side = max(2, int(np.sqrt(n)))
    g_und = nx.grid_2d_graph(side, side)
    mapping = {coord: i for i, coord in enumerate(g_und.nodes())}
    g_und = nx.relabel_nodes(g_und, mapping)
    dg = nx.DiGraph(); dg.add_nodes_from(range(side * side))
    for u, v in g_und.edges():
        dg.add_edge(u, v, weight=1, capacity=base_cap)
        dg.add_edge(v, u, weight=1, capacity=base_cap)
    return dg, side * side


def write_te_format(dg, out_path):
    num_nodes = dg.number_of_nodes()
    edges = list(dg.edges(data=True))
    num_links = len(edges)
    with open(out_path, 'w') as f:
        f.write(f'Node: {num_nodes}\tLink: {num_links}\n')
        f.write('link_idx\tsrc\tdst\tweight\tcapacity\n')
        for idx, (u, v, data) in enumerate(edges):
            w = int(data.get('weight', 1))
            cap = float(data.get('capacity', 1000.0))
            f.write(f'{idx}\t{u}\t{v}\t{w}\t{cap}\n')
    print(f'[+] 拓扑: {out_path}')
    print(f'    {num_nodes} nodes, {num_links} links, '
          f'avg degree = {2*num_links/num_nodes:.2f}')


# ============================================================
#  流量生成
# ============================================================
def gravity_lognormal(n, total_kbps, alpha=1.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.pareto(a=alpha, size=n) + 1.0
    T = np.outer(P, P)
    np.fill_diagonal(T, 0)
    T = T / T.sum() * total_kbps
    eps = np.exp(rng.normal(0.0, noise, size=(n, n)))
    np.fill_diagonal(eps, 1.0)
    T = T * eps
    T = T / T.sum() * total_kbps
    return T


def gravity(n, total_kbps, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.pareto(a=alpha, size=n) + 1.0
    T = np.outer(P, P)
    np.fill_diagonal(T, 0)
    T = T / T.sum() * total_kbps
    return T


def bimodal(n, total_kbps, frac_busy=0.3, ratio=20.0, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < frac_busy
    np.fill_diagonal(mask, False)
    base = rng.exponential(1.0, size=(n, n)) * mask * ratio \
         + rng.exponential(1.0, size=(n, n)) * (~mask)
    np.fill_diagonal(base, 0)
    base = base / base.sum() * total_kbps
    return base


def write_tm(matrices, out_path, scale=100):
    file_factor = 375.0  # file_value = real_kbps * 375 (scale=100)
    with open(out_path, 'w') as f:
        for i, mat in enumerate(matrices):
            np.fill_diagonal(mat, 0)
            file_mat = mat * file_factor
            line = ' '.join(f'{v:.1f}' for v in file_mat.flatten())
            f.write(line + '\n')
            if (i + 1) % 100 == 0:
                print(f'    wrote {i+1}/{len(matrices)}')
    print(f'[+] 流量: {out_path} ({len(matrices)} TM, '
          f'{os.path.getsize(out_path)/1024/1024:.1f} MB)')


# ============================================================
#  入口
# ============================================================
def main():
    p = argparse.ArgumentParser()
    # 拓扑
    p.add_argument('--nodes', type=int, default=200)
    p.add_argument('--model', default='waxman', choices=['waxman', 'ba', 'grid'])
    p.add_argument('--alpha', type=float, default=0.3)
    p.add_argument('--beta', type=float, default=0.4)
    p.add_argument('--ba-m', type=int, default=3)
    p.add_argument('--base-cap', type=float, default=1000.0)
    # 流量
    p.add_argument('--num-train', type=int, default=800)
    p.add_argument('--num-test', type=int, default=200)
    p.add_argument('--total-kbps', type=float, default=5e7)
    p.add_argument('--traffic-model', default='gravity_ln',
                   choices=['gravity', 'gravity_ln', 'bimodal'])
    p.add_argument('--noise', type=float, default=0.5)
    p.add_argument('--frac-busy', type=float, default=0.3)
    p.add_argument('--busy-ratio', type=float, default=20.0)
    p.add_argument('--scale', type=float, default=100)
    # 输出
    p.add_argument('--out-dir', default='data')
    p.add_argument('--prefix', default=None,
                   help='文件名前缀,默认 large<nodes>')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true', help='覆盖已存在文件')
    args = p.parse_args()

    if args.prefix is None:
        args.prefix = f'large{args.nodes}'

    topo_path = os.path.join(args.out_dir, f'{args.prefix}_topo')
    train_path = os.path.join(args.out_dir, f'{args.prefix}_train.txt')
    test_path = os.path.join(args.out_dir, f'{args.prefix}_test.txt')
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 拓扑 ----
    if os.path.exists(topo_path) and not args.force:
        print(f'[*] 拓扑已存在,跳过: {topo_path} (加 --force 覆盖)')
    else:
        print(f'[*] 生成拓扑: model={args.model}, n={args.nodes}')
        if args.model == 'waxman':
            dg = build_waxman(args.nodes, args.alpha, args.beta, args.seed, args.base_cap)
            real_n = args.nodes
        elif args.model == 'ba':
            dg = build_ba(args.nodes, args.ba_m, args.seed, args.base_cap)
            real_n = args.nodes
        else:
            dg, real_n = build_grid(args.nodes, args.seed, args.base_cap)
            print(f'    grid 实际节点数 = {real_n} (side={int(real_n**0.5)})')
            # grid 模式下,后续流量也用 real_n
            args.nodes = real_n
            topo_path = os.path.join(args.out_dir, f'{args.prefix}_topo')
            train_path = os.path.join(args.out_dir, f'{args.prefix}_train.txt')
            test_path = os.path.join(args.out_dir, f'{args.prefix}_test.txt')
        write_te_format(dg, topo_path)

    # ---- 流量 ----
    need_train = args.force or not os.path.exists(train_path)
    need_test = args.force or not os.path.exists(test_path)

    if not need_train and not need_test:
        print(f'[*] 流量已存在,跳过: {train_path}, {test_path}')
    else:
        print(f'[*] 生成流量: model={args.traffic_model}, '
              f'train={args.num_train}, test={args.num_test}')
        rng = np.random.default_rng(args.seed)
        all_tms = []
        for i in range(args.num_train + args.num_test):
            cur_total = args.total_kbps * (0.7 + 0.6 * rng.random())
            if args.traffic_model == 'gravity':
                T = gravity(args.nodes, cur_total, args.alpha, seed=args.seed + i)
            elif args.traffic_model == 'gravity_ln':
                T = gravity_lognormal(args.nodes, cur_total, args.alpha,
                                      args.noise, seed=args.seed + i)
            else:
                T = bimodal(args.nodes, cur_total, args.frac_busy,
                            args.busy_ratio, seed=args.seed + i)
            all_tms.append(T.astype(np.float32))
        if need_train:
            write_tm(all_tms[:args.num_train], train_path, args.scale)
        if need_test:
            write_tm(all_tms[args.num_train:], test_path, args.scale)

    # ---- 摘要 ----
    print('\n' + '=' * 60)
    print('生成完成。可被 train_direct 直接加载:')
    print('=' * 60)
    print(f'    Topology("{topo_path}", max_paths_per_pair=8)')
    print(f'    TrafficLoader("{train_path}", topo.num_nodes)')
    print(f'    TrafficLoader("{test_path}", topo.num_nodes)')


if __name__ == '__main__':
    main()
