import platform

import pandas as pd


def main():
    print("Hello from uv-docker!")
    print(f"Python: {platform.python_version()}")
    print(f"pandas: {pd.__version__}")


if __name__ == "__main__":
    main()
