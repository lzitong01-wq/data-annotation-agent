# 一致性指标说明

## 为什么不能只看「一致率」

两个标注者标了 100 条，95 条一致，看起来很好。但如果 90 条都属于同一个类别，那么**闭着眼睛全猜这个类别**也能拿到 90% 的一致率。原始一致率没有扣除「碰巧一致」的部分。

κ 类指标做的就是这件事：把观察到的分歧，跟「纯随机情况下预期的分歧」相比。

## Cohen's κ（两个标注者）

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

- $P_o$：观察一致率，即两人标签相同的比例
- $P_e$：期望一致率，由各自标签的边缘分布相乘求和得到

实现见 `quality/agreement.py::cohens_kappa`。

```python
from data_annotation_agent.quality.agreement import cohens_kappa
cohens_kappa(["A", "B", "A"], ["A", "B", "A"])   # 1.0
cohens_kappa(["A", "B"], ["B", "A"])             # 负数
```

取值 $[-1, 1]$。1 为完全一致，0 为与随机猜测无异，负数为系统性相反。

## Fleiss' κ（三个及以上标注者）

Cohen's κ 只能处理两方，而且要求两方标的是同一批样本。Fleiss' κ 把它推广到任意数量的标注者：

$$\kappa = \frac{\bar{P} - \bar{P_e}}{1 - \bar{P_e}}$$

其中 $\bar{P}$ 是各样本一致程度的平均，$\bar{P_e}$ 是各类别总体比例的平方和。

实现见 `fleiss_kappa`。当标注者数 = 2 时，它与 Cohen's κ 数值接近但不完全相等（边缘分布的处理方式不同）。

## Krippendorff's α

理论上最灵活：支持任意数量的标注者、任意测量尺度（名义 / 顺序 / 区间）、允许缺失值。

**本项目的实现是简化版**：对名义数据直接复用 Fleiss' κ，并且会把含缺失值（`None`）的样本整行丢弃。

```python
krippendorff_alpha([["A", "A"], [None, "B"]])   # 第二行被丢弃
```

如果你需要严格的 α 值（例如论文里要报这个数），请用官方库替换：

```bash
pip install krippendorff
```

```python
import krippendorff
alpha = krippendorff.alpha(reliability_data=ratings, level_of_measurement="nominal")
```

## 强度解读（Landis & Koch）

`interpret_kappa()` 采用临床流行病学最常用的分档：

| κ 区间 | 解读 |
| --- | --- |
| < 0 | 无一致性 |
| 0.00 – 0.20 | 微弱一致性 |
| 0.21 – 0.40 | 一般一致性 |
| 0.41 – 0.60 | 中等一致性 |
| 0.61 – 0.80 | 显著一致性 |
| 0.81 – 1.00 | 几乎完美一致性 |

注意这是**约定俗成**，不是自然定律。标注任务越难（如细粒度情感、医学实体边界），越难拿到高分；0.6 在某些任务上已经算可用。

## 该用哪个

| 场景 | 建议 |
| --- | --- |
| 两个模型互检 / 模型 vs 人工 | Cohen's κ（`InspectorPreset.LIGHT`） |
| 三个及以上模型互检 | Fleiss' κ（`InspectorPreset.STANDARD` / `DEEP`） |
| 有缺失标注、或需要报论文指标 | 换官方 `krippendorff` 库 |

## 与停止条件的关系

`MonitorAgent` 的保守 / 平衡模式直接用 κ 当停止阈值：

| 预设 | 阈值 | 附加条件 |
| --- | --- | --- |
| 保守 `conservative` | κ ≥ 0.8 | 最近 3 次都达标 |
| 平衡 `balanced` | κ ≥ 0.6 | 最近 3 次都达标 |
| 宽松 `lenient` | 不看 κ | 标完预设数量即停 |
| 手动 `manual` | 不看 κ | 由使用者决定 |

「稳定」的判定是最近 `window=3` 次记录全部达标，避免单次波动误判收敛。

## 一个容易踩的坑

多 AI 互检时，如果所有模型用的是同一个后端（同一个模型、同一份 prompt、temperature=0），κ 会接近 1.0，但这个高分毫无信息量——它只说明「同一个模型问了两次答案一样」。

真正有意义的互检需要**模型之间存在差异**：不同厂商、不同规模、不同 temperature。mock 模式为了让流程可演示，会用 30% 概率注入受控分歧，这也是为什么模拟模式下 κ 常在 0.5 附近。
