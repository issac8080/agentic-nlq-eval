Write the answer from the question, the executed SQL, and the result rows only.
Do not add facts that are not in those rows or in the SQL.
Return JSON only:
{"answer_values": [row objects], "entities": ["names mentioned"], "narrative": "short explanation", "evidence": [{"claim_text": "...", "source": "result_row", "ref": "row 0"}]}
source must be "result_row" or "sql".

Question:
<<QUESTION>>

SQL:
<<SQL>>

Row count: <<ROW_COUNT>>
Rows (capped):
<<ROWS>>

Revision notes, if any:
<<ERRORS>>
