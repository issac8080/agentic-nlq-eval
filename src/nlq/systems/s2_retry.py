"""S2: S1 plus an error-driven retry. The cap matches S3: ``max_sql_retries`` extra attempts."""

from __future__ import annotations

import time
from typing import Any

from nlq.benchmark.schema import BenchmarkQuestion
from nlq.data.schema_context import SchemaContext, render_schema_context
from nlq.llm.base import LLMClient, LLMResult
from nlq.llm.prompts import PromptLibrary
from nlq.logging_utils import JsonlRunLogger
from nlq.systems.answer import SystemResult, assemble_result, generate_sql, narrate, run_sql, validation_for
from nlq.systems.sql_validation import SqlValidation


class RetrySystem:
    """Feed validator and execution errors back to the model, up to the shared retry cap."""

    name = "s2"

    def __init__(
        self,
        llm: LLMClient,
        engine: Any,
        schema: SchemaContext,
        prompts: PromptLibrary,
        *,
        max_sql_retries: int = 3,
        row_cap: int = 200,
        logger: JsonlRunLogger | None = None,
        run_id: str = "s2",
    ) -> None:
        self.llm = llm
        self.engine = engine
        self.schema = schema
        self.prompts = prompts
        self.max_sql_retries = max_sql_retries
        self.row_cap = row_cap
        self.logger = logger
        self.run_id = run_id
        self.schema_text = render_schema_context(schema)

    def answer(self, question: BenchmarkQuestion) -> SystemResult:
        started = time.perf_counter()
        calls: list[LLMResult] = []
        errors: list[str] = []
        sql = ""
        validation = SqlValidation(passed=False, messages=[])
        query = None
        exec_time = 0.0
        retries = 0
        for attempt in range(self.max_sql_retries + 1):
            if attempt:
                retries = attempt
            try:
                sql, generated = generate_sql(
                    self.llm,
                    self.prompts,
                    self.schema_text,
                    question.question,
                    errors,
                    logger=self.logger,
                    run_id=self.run_id,
                    question_id=question.id,
                )
                calls.append(generated)
            except Exception as exc:
                errors.append(str(exc))
                continue
            validation = validation_for(sql, self.schema)
            if not validation.passed:
                errors.extend(validation.messages)
                continue
            query, error, spent = run_sql(self.engine, sql, self.row_cap)
            exec_time += spent
            if error is not None or query is None:
                errors.append(error or "execution failed")
                query = None
                continue
            break
        else:
            return assemble_result(
                question=question,
                schema=self.schema,
                sql=sql,
                validation=SqlValidation(passed=False, messages=errors),
                query=None,
                narrative_payload=None,
                llm_results=calls,
                exec_time_s=exec_time,
                started=started,
                retries=retries,
                failed=True,
            )
        if query is None:
            return assemble_result(
                question=question,
                schema=self.schema,
                sql=sql,
                validation=SqlValidation(passed=False, messages=errors or ["no result"]),
                query=None,
                narrative_payload=None,
                llm_results=calls,
                exec_time_s=exec_time,
                started=started,
                retries=retries,
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
            return assemble_result(
                question=question,
                schema=self.schema,
                sql=sql,
                validation=SqlValidation(passed=False, messages=[str(exc)]),
                query=query,
                narrative_payload=None,
                llm_results=calls,
                exec_time_s=exec_time,
                started=started,
                retries=retries,
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
            retries=retries,
            failed=False,
        )
