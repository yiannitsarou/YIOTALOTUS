# -*- coding: utf-8 -*-
from step2_finalize import export_step2_nextcol_full
export_step2_nextcol_full(
    step1_workbook_path="STEP1_IMMUTABLE_MULTISHEET_NODUP (6).xlsx",
    out_xlsx_path="STEP2_NEXTCOL_FULL.xlsx",
    seed=42,
    max_results=5,
    sheet_naming="ΣΕΝΑΡΙΟ_{id}",
)
print("OK: Δημιουργήθηκε το STEP2_NEXTCOL_FULL.xlsx")
