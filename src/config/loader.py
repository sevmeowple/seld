from pathlib import Path
from typing import Any, TypeVar, Type

import tomli as tomli

from config.schema import Config

T = TypeVar('T')


def _find_project_root() -> Path:
    """向上查找 pyproject.toml 或 .git 所在目录作为项目根目录"""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # 如果找不到标记，返回当前工作目录（避免崩溃）
    return current


def _resolve_project_path(path: str | Path) -> Path:
    """将路径解析为绝对路径。如果是相对路径，则相对于项目根目录。"""
    path = Path(path)
    if path.is_absolute():
        return path
    root = _find_project_root()
    return (root / path).resolve()


def load_toml(path: str | Path) -> dict[str, Any]:
    # 自动相对于项目根目录解析
    resolved_path = _resolve_project_path(path)
    with open(resolved_path, "rb") as f:
        return tomli.load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典，override 优先级更高"""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config_generic(
    config_class: Type[T],
    base_path: str | Path = "configs/base.toml",
    local_path: str | Path = "configs/local.toml",
    **cli_overrides
) -> T:
    """Generic config loader that supports any config class"""
    # 1. 解析路径（相对于项目根目录）
    base_path = _resolve_project_path(base_path)
    local_path = _resolve_project_path(local_path)

    # 2. 加载基础配置（会自动处理相对路径）
    config_dict = load_toml(base_path)

    # 3. 合并本地配置（gitignore，个人开发设置）
    if local_path.exists():
        local_dict = load_toml(local_path)
        config_dict = deep_merge(config_dict, local_dict)

    # 4. 合并命令行覆盖（最高优先级）
    config_dict = deep_merge(config_dict, cli_overrides)

    # 5. Pydantic 验证
    return config_class(**config_dict)


def load_config(
    base_path: str | Path = "configs/base.toml",
    local_path: str | Path = "configs/local.toml",
    **cli_overrides
) -> Config:
    """Load standard Config"""
    return load_config_generic(Config, base_path, local_path, **cli_overrides)