# 贡献指南

## 环境准备

```bash
git clone https://github.com/lzitong01-wq/data-annotation-agent.git
cd data-annotation-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

不需要 API Key：所有测试与示例都能在 `mock` 后端下跑通。

## 常用命令

```bash
pytest -q                        # 全量测试
pytest tests/test_agreement.py   # 单个文件
pytest -k kappa                  # 按名字筛选
pytest --cov                     # 覆盖率

ruff check src tests             # 静态检查
ruff check --fix src tests       # 自动修复

python examples/quickstart.py    # 冒烟测试
python -m data_annotation_agent  # 启动界面
```

## 目录约定

| 目录 | 放什么 |
| --- | --- |
| `src/data_annotation_agent/core/` | 数据模型、流程编排、导出 |
| `src/data_annotation_agent/agents/` | 各角色 Agent，一个文件一个角色 |
| `src/data_annotation_agent/llm/` | LLM 后端适配 |
| `src/data_annotation_agent/quality/` | 一致性指标、监控与成本 |
| `src/data_annotation_agent/ui/` | Gradio 界面（唯一依赖 gradio 的地方） |
| `tests/` | pytest 用例，按被测模块命名 `test_<module>.py` |
| `examples/` | 可直接运行的示例脚本 |
| `docs/` | 架构、扩展、指标说明 |
| `data/` | 本地测试数据，不提交实际内容 |

## 代码风格

- 行宽 120，目标 Python 3.10+
- 公开函数写 docstring，说明「做什么」而不是「怎么做的」
- 中文注释可以，但不要用注释复述代码
- 新增模块后记得在 `pyproject.toml` 无需改动（`packages.find` 自动发现），但要确认能 `python -c "import data_annotation_agent.<新模块>"`

## 测试要求

- 新功能必须带测试，且**使用真实断言**（`assert`），不要写只打印结果的脚本式用例
- 涉及随机性的逻辑（如 mock 模式的分歧注入）不要断言具体标签，只断言数量、结构或不变量
- 提交前跑 `pytest -q`，CI 会在 Python 3.10 / 3.11 / 3.12 上重复验证

## 提交信息

参考 Conventional Commits：

```
feat: 新增图像任务的界面输入
fix: 修正 κ 计算中空输入返回 None
docs: 补充多模态扩展说明
refactor: 拆分 exporter 的格式化逻辑
test: 补充 reviewer 去重用例
chore: 升级 gradio 到 6.1
```

## 提交 PR

1. 从 `main` 拉分支：`git checkout -b feat/your-feature`
2. 改完跑通 `pytest -q` 与 `ruff check src tests`
3. PR 描述里写清：解决什么问题、怎么验证的、有没有破坏性变更
4. 涉及接口变更请同步更新 `docs/` 与 `CHANGELOG.md`

## 安全

- **不要提交 API Key**。代码里只从运行时参数或环境变量读取（`LLM_API_KEY` 等）
- 界面没有鉴权，`launch()` 默认只监听 `127.0.0.1`。改成 `0.0.0.0` 前请确认网络环境可信
- 发现安全问题请通过 issue 私下沟通，不要在公开 issue 里贴可利用细节
