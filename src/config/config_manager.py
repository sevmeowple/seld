import argparse
from pathlib import Path

from config.loader import load_config
from config.schema import Config

_config: Config | None = None


def init_config(**kwargs) -> Config:
    """应用启动时调用一次"""
    global _config
    _config = load_config(**kwargs)
    return _config


def get_config() -> Config:
    """业务代码中随处调用"""
    if _config is None:
        raise RuntimeError("Config not initialized. Call init_config() first")
    return _config

def C() -> Config:  
    return get_config()


def parse_cli_args() -> dict:
    """极简实现：--key.subkey=value"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    parser.add_argument("overrides", nargs="*", help="格式：section.key=value")

    args = parser.parse_args()

    # 解析 overrides
    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}, expected key=value")
        key_path, val = item.split("=", 1)

        # 构建嵌套字典：a.b.c=1 -> {"a": {"b": {"c": 1}}}
        keys = key_path.split(".")
        current = overrides
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        # 尝试自动类型转换（int/float/bool/str）
        current[keys[-1]] = _auto_cast(val)

    return {"base_path": args.config, **overrides}


def _auto_cast(val: str):
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val
