"""S1: one SQL generation, then execution, then narration. No retry."""

from __future__ import annotations

import time
from typing import Any

from nlq.benchmark.schema import BenchmarkQuestion
from nlq.data.schema_context import SchemaContext, render_schema_context
from nlq.llm.base import LLMClient
from nlq.llm.prompts import PromptLibrary
from nlq.logging_utils import JsonlRunLogger
from nlq.systems.answer import (
    SystemResult,
    assemble_result,
    generate_sql,
    narrate,
    run_sql,
    validation_for,
)
from nlq.systems.sql_validation import SqlValidation


class SinglePassSystem:
    """Text-to-SQL in one model call. The narration prompt is the shared skeleton."""

    name = "s1"

    def __init__(
        self,
        llm: LLMClient,
        engine: Any,
        schema: SchemaContext,
        prompts: PromptLibrary,
        *,
        row_cap: int = 200,
        logger: JsonlRunLogger | None = None,
        run_id: str = "s1",
    ) -> None:
        self.llm = llm
        self.engine = engine
        self.schema = schema
        self.prompts = prompts
        self.row_cap = row_cap
        self.logger = logger
        self.run_id = run_id
        self.schema_text = render_schema_context(schema)

    def answer(self, question: BenchmarkQuestion) -> SystemResult:
        started = time.perf_counter()
        calls = []
        try:
            sql, generated = generate_sql(
                self.llm,
                self.prompts,
                self.schema_text,
                question.question,
                [],
                logger=self.logger,
                run_id=self.run_id,
                question_id=question.id,
            )
            calls.append(generated)
        except Exception as exc:
            return assemble_result(
                question=question,
                schema=self.schema,
                sql="",
                validation=SqlValidation(passed=False, messages=[str(exc)]),
                query=None,
                narrative_payload=None,
                llm_results=calls,
                exec_time_s=0,
                started=started,
                retries=0,
                failed=True,
            )
        validation = validation_for(sql, self.schema)
        if not validation.passed and any("Only a SELECT" in message or "DDL" in message for message in validation.messages):
            return assemble_result(
                question=question,
                schema=self.schema,
                sql=sql,
                validation=validation,
                query=None,
                narrative_payload=None,
                llm_results=calls,
                exec_time_s=0,
                started=started,
                retries=0,
                failed=True,
            )
        query, error, exec_time = run_sql(self.engine, sql, self.row_cap)
        if error is not None or query is None:
            return assemble_result(
                question=question,
                schema=self.schema,
                sql=sql,
                validation=SqlValidation(passed=False, messages=[error or "execution failed"]),
                query=None,
                narrative_payload=None,
                llm_results=calls,
                exec_time_s=exec_time,
                started=started,
                retries=0,
                failed=True,
            )
        try:
            payload, narrated = narrate(
                self.llm,
                self.prompts,
                question.question,
                sql,
                query,
                [],
                logger=self.logger,
                run_id=self.run_id,
                question_id=question.id,
            )
            calls.append(narrated)
        except Exception as exc:
            payload = {"narrative": "", "answer_values": [], "entities": []}
            return assemble_result(
                question=question,
                schema=self.schema,
                sql=sql,
                validation=SqlValidation(passed=False, messages=[str(exc)]),
                query=query,
                narrative_payload=payload,
                llm_results=calls,
                exec_time_s=exec_time,
                started=started,
                retries=0,
                failed=True,
            )
        return assemble_result(
            question=question,
            schema=self.schema,
            sql=sql,
            validation=validation,
            query=query,
            narrative_payload=payload,
            llm_results=calls,
            exec_time_s=exec_time,
            started=started,
            retries=0,
            failed=False,
        )
