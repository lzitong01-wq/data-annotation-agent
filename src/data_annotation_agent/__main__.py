"""命令行入口：``python -m data_annotation_agent`` 启动 Web 界面。"""

import sys

from data_annotation_agent.ui.app import main

if __name__ == "__main__":
    sys.exit(main() or 0)
