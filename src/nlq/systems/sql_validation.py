"""Deterministic Spark SQL checks. No language model is involved.

The validator rejects anything that is not a single read-only query, any table
or column outside the business schema, and joins that have no condition.
Those failures are safe to show back to the model; they do not require executing SQL.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp, parse

READ_ONLY = (exp.Select, exp.Union, exp.Intersect, exp.Except)


class SqlValidation(BaseModel):
    """Structured messages the retry loop can paste into the next prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    messages: list[str] = Field(default_factory=list)


def validate_sql(sql: str, schema: dict[str, set[str]]) -> SqlValidation:
    """Parse ``sql`` as Spark SQL and check it against ``schema`` (table -> columns)."""
    messages: list[str] = []
    try:
        statements = [item for item in parse(sql, read="spark") if item is not None]
    except Exception as exc:
        return SqlValidation(passed=False, messages=[f"SQL parse error: {exc}"])
    if len(statements) != 1:
        return SqlValidation(passed=False, messages=["Exactly one SQL statement is allowed"])
    expression = statements[0]
    if not isinstance(expression, READ_ONLY):
        return SqlValidation(passed=False, messages=["Only a SELECT query is allowed; DDL and DML are rejected"])

    cte_names = {cte.alias_or_name for cte in expression.find_all(exp.CTE)}
    aliases = {alias.alias_or_name for alias in expression.find_all(exp.Alias)}
    alias_to_table: dict[str, str] = {}
    referenced: set[str] = set()
    for table in expression.find_all(exp.Table):
        name = table.name
        qualifier = table.alias_or_name
        alias_to_table[qualifier] = name
        if name in cte_names:
            continue
        if name not in schema:
            messages.append(f"Unknown table: {name}")
            continue
        referenced.add(name)

    for column in expression.find_all(exp.Column):
        column_name = column.name
        if column_name == "*":
            continue
        qualifier = column.table
        if qualifier:
            target = alias_to_table.get(qualifier, qualifier)
            if target in cte_names:
                continue
            columns = schema.get(target)
            if columns is None:
                messages.append(f"Unknown table alias: {qualifier}")
            elif column_name not in columns:
                messages.append(f"Unknown column: {qualifier}.{column_name}")
            continue
        if column_name in aliases:
            continue
        owners = [table for table in referenced if column_name in schema.get(table, set())]
        if not owners:
            messages.append(f"Unknown column: {column_name}")
        elif len(owners) > 1:
            messages.append(f"Ambiguous column: {column_name}")

    for join in expression.find_all(exp.Join):
        kind = (join.args.get("kind") or "").upper()
        has_condition = join.args.get("on") is not None or bool(join.args.get("using"))
        if kind == "CROSS" or not has_condition:
            messages.append("Cross join without a condition is not allowed")

    unique = list(dict.fromkeys(messages))
    return SqlValidation(passed=not unique, messages=unique)


def extract_tables(sql: str, schema_names: set[str]) -> list[str]:
    """Base tables the statement reads, ignoring CTE names."""
    try:
        expression = parse(sql, read="spark")[0]
    except Exception:
        return []
    if expression is None:
        return []
    cte_names = {cte.alias_or_name for cte in expression.find_all(exp.CTE)}
    names = []
    for table in expression.find_all(exp.Table):
        if table.name in schema_names and table.name not in cte_names and table.name not in names:
            names.append(table.name)
    return names


def extract_filters(sql: str) -> list[str]:
    """WHERE and HAVING text, so provenance can be checked against the executed SQL."""
    try:
        expression = parse(sql, read="spark")[0]
    except Exception:
        return []
    if expression is None:
        return []
    filters: list[str] = []
    for node in expression.find_all(exp.Where, exp.Having):
        rendered = node.this.sql(dialect="spark")
        if rendered not in filters:
            filters.append(rendered)
    return filters
