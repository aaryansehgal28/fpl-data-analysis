from __future__ import annotations
import pandas as pd

class DataQualityError(ValueError): pass
def require_columns(df: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing: raise DataQualityError(f"{table}: missing columns {sorted(missing)}")
def require_unique(df: pd.DataFrame, keys: list[str], table: str) -> None:
    require_columns(df, keys, table)
    if df.duplicated(keys).any(): raise DataQualityError(f"{table}: duplicate key(s) {keys}")
def require_fk(child: pd.DataFrame, child_key: str, parent: pd.DataFrame, parent_key: str, table: str) -> None:
    unknown = set(child[child_key].dropna()) - set(parent[parent_key].dropna())
    if unknown: raise DataQualityError(f"{table}: invalid {child_key}, sample={list(unknown)[:5]}")
def require_nonnegative(df: pd.DataFrame, columns: list[str], table: str) -> None:
    for col in columns:
        if (pd.to_numeric(df[col], errors="coerce") < 0).any(): raise DataQualityError(f"{table}: negative {col}")
