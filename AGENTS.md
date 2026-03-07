## GUIDELINES 工作总则（先看）

- 优先遵循本文件
- 使用UV管理python环境
  - 如,添加python依赖使用`uv add xxx`
  - 若没有UV环境（不存在pyproject.toml文件），先uv init创建uv项目
- 本机器配置为RTX4060显卡 13th Gen Intel(R) Core(TM) i9-13900H
- 对跨模块变更（环境注册、算法接入、配置项）要保持“代码 + 配置 + 文档”同步。
- 代码修改**不要进行commit**
- 每次工作可以修改的文件**仅限项目目录内**
- 每次工作**基本流程**
  - 阅读该文档
  - 根据AGENTS.md文档确定自己任务下一步需要看哪个部分并仔细阅读
  - 进行修改
  - 执行必要的测试
  - 直到达到目标
  - **在完成目标后**，将以上过程记录在一个md文件内，命名规则<日期>_<session-id>_<概括任务信息>，放在**report/目录**下
    - 并且生成一份给其他agent读，方便你的其他“同事”了解工作情况的文档，放在**report/coop/目录**下，命名规则同上
- 同一个会话中一般有多次工作，每次工作**如有必要的话**要**更新**对应的report和coop下的文件，如：1. 做了新的修改，在report中的内容补充对当次工作的描述；2. coop下的内容经过当次工作修改后已不适用，需要修改；......
  - 但是，即使在同一个会话中，如果当次工作（指的是用户提出一次要求后Agent做的一系列工作）的主题和已经存在的report文件**有明显不同**，要问用户是否要需要创建新的report文件
- 新增：report需要添加序号，同一日期需要从0开始累加，**新**命名规则<日期>_<序号>_<概括任务信息>

## INIT（每次新会话先做）

- 先执行环境初始化检查（推荐在项目根目录）：
  - `python3 -V`
  - `uv --version`
  - `uv venv --python 3.10 .venv`
  - `uv pip install -r script/requirements.txt`
- 资产路径初始化（RoboTwin运行前必须做一次）：
  - `python script/update_embodiment_config_path.py`
- 若需要采集数据，优先直接调用：
  - `python script/collect_data.py <task_name> <task_config>`
  - 说明：`collect_data.sh` 中引用了 `script/.update_path.sh`，当前仓库没有该文件。

## 执行顺序建议（策略算法）

- 标准流程：`数据采集 -> 数据预处理 -> 训练 -> 部署评测 -> 结果记录`
- ACT：`policy/ACT/process_data.sh -> policy/ACT/train.sh -> policy/ACT/eval.sh`
- DP：`policy/DP/process_data.sh -> policy/DP/train.sh -> policy/DP/eval.sh`
- 训练与评测默认优先单卡，按笔记本资源逐步增加开销。

## 开销控制（笔记本优先）

- 首次跑通建议：
  - 任务配置使用 `task_config/demo_clean.yml`
  - 先把 `episode_num` 调小（如 2~5）
  - `render_freq: 0`
- ACT/DP优先于大型VLA（OpenVLA/TinyVLA/DexVLA），先保证端到端流程可复现。
- 发生OOM时优先降：`batch_size`、数据量、图像输入规模；最后再考虑换算法。
