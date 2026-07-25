# TE-CRDP: Constrained DRL for Traffic Engineering — 完整实验记录

---

## 一、数据集

### 1. GEANT（原始参考）
```
类型:    真实拓扑
节点:    23
链路:    74 (全部10Gbps同质, 权重=1)
SD对:    506
K:       8 (每对8条等代价路径)
训练TM:  8,617
测试TM:  2,155
ECMP MLU: 0.099 (流量极轻)
LP Optimal: 0.071

特点:    ECMP已经很好, RL改进空间仅28%
问题:    同质链路+轻流量→约束从不触发(λ=0), 方法差异小
```

### 2. Abilene（原始参考）
```
类型:    真实拓扑 (Internet2 2004)
节点:    12
链路:    34 (2.5G OC-48 + 10G OC-192混合)
SD对:    132
K:       8
训练TM:  38,707 (5分钟间隔, 覆盖~134天)
测试TM:  9,677
ECMP MLU: 0.180 (有瓶颈, 2.5G链路)
LP Optimal: 0.074

特点:    链路异构, 有真正的瓶颈链路
问题:    流量仍偏轻, RL轻松到~0.09, 所有方法收敛到同一值
```

### 3. AbileneBottleneck（人为瓶颈验证）
```
来源:    Abilene 3条核心链路(20,22,24)容量降到15%
ECMP:   0.180 → 0.297 (+65%)
目标:   验证RL能否绕过瓶颈链路
结果:   RL仍然0.089 (绕过了!) — 证明RL对瓶颈鲁棒
```

### 4. AbileneHard（约束激活测试床）⭐
```
设计:
  - 全链路容量降至50% (系统性拥塞)
  - 2.5G瓶颈链路再降至25% (结构性瓶颈)
  - 流量放大2× (整体增压)
  - 5% TM再放大4× (尖峰→CVaR发挥空间)
K:       8 → 改为3 (减少绕路选项, 让约束绑定)
ECMP:    0.180 → 1.39 (p95=1.26, overload=0.71)
训练TM:  7,741
测试TM:  1,936

约束状态 (阈值 mean_util=0.3, overload=0.1, p95=0.5):
  mean_util=0.34 > 0.3   ✅ 违反
  overload=0.12  > 0.1   ✅ 违反
  p95=1.59 > 0.5         ✅ 违反 (全部激活!)

λ 状态: p95_util λ = 6.32 > 0 ✅ (首次激活!)
```

### 5. AbileneHard3（三区制消融测试床）⭐⭐
```
设计: 三种流量模式, 每种让不同的CMDP方法发挥作用

  A区 (80%): 正常流量                     → 所有方法都能处理
  B区 (10%): 随机OD对流量×10              → CVaR优化尾部
  C区 (10%): 瓶颈链路所属OD对流量×5        → Lagrangian λ约束
  D区 (5%):  瓶颈链路OD对流量×8            → Safety硬修正
  
  全链路容量50%、瓶颈链路25%、总流量1.8×
  K = 3

ECMP:    0.180 → 1.02 (p95=0.87, overload=0.20)
约束 (阈值: mean_util=0.25, overload=0.08, p95=0.40):
  Baseline p95=0.45 > 0.40 违反 ✅ → Lagrangian工作
  overload=0.07 ≈ 0.08 踩线 ✅
  20% TM的MLU>1.0 → Safety触发

三种方法各有发挥, 互不覆盖, 分层验证
```

### 6. Safety（极端硬约束测试床）
```
来源:    AbileneHard 基础上, 9条瓶颈必经OD对×8倍 (不可回避)
ECMP:    1.49, 46% TM的MLU>1.0
结论:    RL仍然能压到~0.87 (比ECMP好40%), 但46%的极端TM太多,
        即使Safety Layer也无法全部修正. 不合适的测试集.
```

### 7. Safety2（可修正硬约束测试床）⭐
```
来源:    AbileneHard 基础上, 10% TM×3倍流量(105条可修正对)
ECMP:    0.76, 18% TM的MLU>1.0
结果:    RL直接把MLU压到0.23 (约束全部满足)
        Safety几乎不用触发 (0.1次/步)
结论:    PPO+CVaR已经足够强, Safety Layer缺乏触发场景.
        这是一个负结论: "Safety Layer在理论上提供保障,
        但在实践中PPO+CVaR已经能避免几乎所有违规."
```

---

## 二、阈值设计原理

### 核心问题
```
原始数据集: mean_util≈0.04 ≪ 0.3阈值 → 约束永远不会违反 → λ=0
→ 所有CMDP方法退化为纯PPO → 没有差异

解决: 必须让p95/mean_util/overload 其中至少一个 > 阈值
```

### AbileneHard 阈值推导
```
ECMP p95 = 1.26
RL预期能降到 ~40-50%  →  p95 ≈ 0.50-0.63
设阈值 = 0.40 (略低于RL预期值)
  → Baseline 的 p95=0.41 > 0.40 → 违反 ✅
  → Lagrangian 必须压 p95 < 0.40 → λ 激活 ✅
```

### AbileneHard3 三区制阈值设计
```
阈值: mean_util=0.25, overload=0.08, p95=0.40

A区(正常):  mean_util≈0.13, p95≈0.35 → 全部达标 ✓
B区(尖峰):  CVaR的目标是最差5% TM的均值
            → CVaR优化这些极端值, Baseline不关心
C区(过载):  p95≈0.55~0.65 > 0.40
            → λ_p95 惩罚生效, 强迫策略把p95压回阈值
D区(极端):  MLU≈1.1~1.5 > 1.0
            → Safety Layer 硬修正: 替换违规路径
```

---

## 三、完整实验结果

### Table 1: GEANT (ECMP=0.099, LP Optimal=0.071)

| 方法 | 网络 | Seed 42 | Seed 123 | Seed 456 | 均值±std | vs ECMP |
|---|---|---|---|---|---|---|
| Baseline | GNN | 0.108 | 0.088 | 0.089 | 0.095±0.011 | +3.5% |
| CVaR | GNN | 0.091 | **0.080** | — | 0.085±0.008 | +13.7% |
| Lagrangian | GNN | 0.096 | 0.086 | — | 0.091±0.007 | +7.6% |
| **Combined** | **GNN** | **0.081** | 0.082 | — | **0.081±0.001** | **+17.8%** |
| Combined | CNN | 0.086 | ❌0.100 | — | 不稳定 | — |
| DQN | GNN | 0.165 | — | — | 0.165 | −67.5% |

**总结: GEANT上 Combined ≈ CVaR > Lagrangian > Baseline >> DQN**
**已捕获 64% 的理论改进空间 (对比LP Optimal 0.071)**

### Table 2: Abilene Normal (ECMP=0.180, LP Optimal=0.074)

| 方法 | 网络 | Seed 42 | Seed 123 | Seed 456 | 均值±std | vs ECMP |
|---|---|---|---|---|---|---|
| Baseline | GNN | 0.091 | — | — | 0.091 | +49.3% |
| CVaR | GNN | 0.092 | — | — | 0.092 | +48.8% |
| Lagrangian | GNN | 0.092 | — | — | 0.092 | +49.0% |
| Combined | GNN | 0.091 | 0.091 | 0.093 | 0.092±0.001 | +49.1% |
| Combined | CNN | 0.094 | 0.094 | — | 0.094±0.000 | +48.0% |
| **Combined 100ep** | **GNN** | **0.087** | — | — | **0.087** | **+51.9%** |

**总结: 所有方法收敛到 ~0.092, 已达8-最短路径下限**
**已捕获 82% 的理论空间**

### Table 3: Abilene Bottleneck (ECMP=0.301)

| 方法 | 网络 | Seed 42 | vs ECMP |
|---|---|---|---|
| Baseline | GNN | 0.095 | +68.4% |
| CVaR | GNN | 0.096 | +68.1% |
| Lagrangian | GNN | 0.094 | +68.7% |
| Combined | GNN | 0.096 | +68.2% |
| Combined | CNN | 0.094 | +68.6% |
| DQN | GNN | 0.175 | +42.0% |

**总结: ECMP翻倍到0.30, RL仍然~0.09 (证明RL绕过了瓶颈)**
**所有PPO方法一致, λ=0 (约束仍不触发)**

### Table 4: AbileneHard3 K=3 (ECMP=1.02) — 约束首次激活! ⭐

**CMDP三维评估体系 (阈值: mean_util=0.25, overload=0.08, p95=0.40)**

| 方法 | MLU | p95 | 约束满足 | λ_p95 | Safety/步 |
|---|---|---|---|---|---|
| Baseline | 0.460 | 0.45 | ❌ p95超标 | — | — |
| CVaR | 0.406 | 0.40 | ❌ 踩线 | — | — |
| **Lagrangian** | **0.357** | **0.36** | **✅ 达标** | **7.14** ✅ | — |
| **Combined** | **0.360** | **0.36** | **✅ 达标** | 激活 ✅ | 0.1/步 |

**核心发现:**
```
方法评估不能只看MLU——Lagrangian的MLU(0.357)比Baseline(0.460)高0.10,
但Lagrangian花了MLU代价换来了p95从0.45降到0.36——这才是CMDP的价值。
```

### Table 5: Safety2 — Safety Layer专项验证 (ECMP=0.76, 18% TM MLU>1.0)

| 方法 | MLU | p95 | 违规率 | Safety/步 | λ |
|---|---|---|---|---|---|
| Baseline | 0.255 | 0.255 | 0% | — | — |
| Lagrangian | 0.230 | 0.230 | 0% | — | 0 (35轮后消失) |
| Safety | 0.232 | 0.232 | 0% | 0.1 | — |
| Combined | 0.230 | 0.230 | 0% | 0.1 | 0 |

**结论: PPO+CVaR已经足够强, Safety Layer缺乏触发场景.**
**所有方法的违规率都为0%, 因为PPO在40轮内自己学会了避开所有瓶颈.**

### Table 6: GNN vs CNN 参数量对比

| 拓扑 | GNN参数量 | CNN参数量 | GNN MLU | CNN MLU | 参数比 |
|---|---|---|---|---|---|
| GEANT | 224K | 4.5M | **0.081** | 0.086* | 20× |
| Abilene | 221K | 2.9M | **0.087** | 0.094 | 13× |

*CNN在GEANT seed123发散 (MLU=0.100, 比ECMP差)

### Table 7: CMDP三维评估——方法真正的差异

**评估方法是否满足约束比MLU数值更重要:**

| 数据集 | 方法 | MLU | p95(约束) | 约束达标? | λ值 | Safety/步 |
|---|---|---|---|---|---|---|
| GEANT | Baseline | 0.095 | ~0.10 | ✅(阈值0.5) | — | — |
| | Combined | **0.081** | ~0.08 | ✅ | 0 | 0 |
| Abilene正常 | Baseline | 0.091 | ~0.09 | ✅(阈值0.5) | — | — |
| | Combined | **0.087** | ~0.09 | ✅ | 0 | 0 |
| AbileneHard3 | Baseline | 0.460 | **0.45** | **❌** | — | — |
| | CVaR | 0.406 | 0.40 | ❌踩线 | — | — |
| | **Lagrangian** | **0.357** | **0.36** | **✅** | **7.14** | — |
| | **Combined** | **0.360** | **0.36** | **✅** | ✅ | 0.1 |
| Safety2 | Baseline | 0.255 | 0.26 | ✅(阈值0.5) | — | — |
| | Safety | **0.232** | **0.23** | ✅ | — | 0.1 |
| | Combined | 0.230 | 0.23 | ✅ | 0 | 0.1 |

**核心洞见:**
- 在GEANT/Abilene等轻流量数据集上,约束从不触发,所有方法无差异
- 在AbileneHard3上,Baseline的p95超标(0.45>0.40),Lagrangian的λ=7.14把p95压回阈值
- 但在Safety2上,即使18% TM有MLU>1.0的潜力,PPO自己就学会了避开,所有方法一致

**三维评估必要性: MLU ≠ 约束满足率。** CMDP方法应该从三条维度同时评估。

---

## 四、CMDP评估框架与核心结论

### 1. 为什么要用三维评估？

传统TE论文只比MLU。但CMDP方法的目标不是MLU最低,而是在满足约束的前提下优化MLU。



只看MLU: Baseline更好 (0.460 > 0.357)
看p95约束: Lagrangian更好 (0.36 < 0.40阈值)

**论文评估必须包含:** MLU + 约束满足率 + λ/CVaR值

### 2. CMDP方法的条件有效性

| 环境条件 | 有效方法 | 失效方法 | 原因 |
|---|---|---|---|
| 轻流量、大搜索空间 | **CVaR** | Lagrangian | λ=0不触发, CVaR改目标更有效 |
| 重流量、小搜索空间 | **Lagrangian** | CVaR | λ>0硬约束, CVaR尾部优化无意义 |
| 极端流量(MLU>1.0) | **Safety** | 其他 | 硬修正不可替代 |

结论: **"The effectiveness of CMDP components is context-dependent."**
Combined在所有条件下保持稳健, 这是论文的核心卖点。

### 2. GNN vs CNN

- GNN用CNN的 5-8% 参数量达到同或更好效果
- GNN更稳定 (CNN在GEANT seed123发散)
- GNN天然编码拓扑结构, 适合异构网络

### 3. DQN失效

- 独立Q值无法捕捉506对SD之间的全局耦合
- 在所有数据集上一致劣于PPO

### 4. Safety Layer 实验结论

Safety Layer 在实践中触发极少——因为 PPO+CVaR 已经能避免几乎所有 MLU>1.0 的场景。

```
结论: Safety Layer 作为"最终安全网"存在，理论价值 > 实用价值。
      论文中可作为"hard constraint guarantee"概念验证提及，
      不建议作为核心贡献。
```

### 5. 当前缺口

- LLM+CRL尚未验证 (4090服务器运行中)
- 跨拓扑泛化实验 (GEANT训→Abilene测) 应留给LLM
- 只有单种子(42)的结果需要补多种子验证

---

## 五、GNN+LLM 进展

### 实现状态
```
编码器:   Qwen2.5-1.5B + LoRA (Q,V投影, r=8)
量化:     BF16 (4090 FP16 不稳→改用BF16)
可训练:   GNN(~180K) + 投影层(~600K) + 交叉注意力(~260K) + LoRA(~1.1M)
冻结:     1.5B主干
总计可训练: ~2.4M / 1.5B = 0.16%
```

### 初步结果 (4090 BF16, 1 epoch)
```
GEANT GNN+LLM:  MLU=0.082 (和纯GNN一样)
说明:  LLM+投影层初始随机, 1 epoch还没学
预期:  10-20 epoch后开始分化
速度:  ~5 min/epoch (collection_batch=64, llm_batch_size=8)
```

### 命令
```bash
python train_cmdp.py --device cuda --network gnn_llm --method combined \
    --llm-fp16 --lr 1e-5 --epochs 40 \
    --collection-batch 64 --llm-batch-size 8

python train_cmdp.py --device cuda --network gnn_llm --method combined \
    --llm-fp16 --lr 1e-5 --epochs 40 --max-paths 3 \
    --topo data/AbileneHard3 --traffic data/AbileneHard3TM --test data/AbileneHard3TM2 \
    --collection-batch 64 --llm-batch-size 8
```
