
import pandas as pd
import importlib.util, sys, re, random, datetime as dt
from pathlib import Path

ROOT = Path("/mnt/data")

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def pick_minrule_across_sheets(step1_6_path: str, out_path: str, seed: int = 42):
    m7 = _load("step7_fixed_final", ROOT / "step7_fixed_final.py")
    xls = pd.ExcelFile(step1_6_path)

    candidates = []
    for sheet in xls.sheet_names:
        # ignore summary sheets
        if str(sheet).strip() in {"Σύνοψη"}: 
            continue
        df = pd.read_excel(step1_6_path, sheet_name=sheet)
        scen_cols = [c for c in df.columns if str(c).startswith("ΒΗΜΑ6_ΣΕΝΑΡΙΟ_")]
        if not scen_cols:
            continue
        col = scen_cols[0]
        res = m7.score_one_scenario(df, col)
        candidates.append((sheet, col, res, df))

    if not candidates:
        raise RuntimeError("No Step 6 scenario columns found in any sheet.")

    # If there exists any with zero broken, restrict to those; else keep all.
    zero_broken = [c for c in candidates if c[2]["broken_friendships"] == 0]
    pool = zero_broken if zero_broken else candidates

    # Sort by: total_score → diff_population → diff_gender_total → diff_greek → sheet name
    pool.sort(key=lambda t: (t[2]["total_score"], t[2]["diff_population"], t[2]["diff_gender_total"], t[2]["diff_greek"], t[0]))

    # Take the best; if multiple tie perfectly, choose one randomly (last tie-breaker)
    best_total = pool[0][2]["total_score"]
    best_dp    = pool[0][2]["diff_population"]
    best_dg    = pool[0][2]["diff_gender_total"]
    best_gr    = pool[0][2]["diff_greek"]
    ties = [t for t in pool if (t[2]["total_score"], t[2]["diff_population"], t[2]["diff_gender_total"], t[2]["diff_greek"]) == (best_total, best_dp, best_dg, best_gr)]
    random.seed(seed)
    chosen_sheet, chosen_col, chosen_score, chosen_df = random.choice(ties)

    # Save FINAL workbook with per-class sheets
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        chosen_df.to_excel(w, index=False, sheet_name="FINAL_SCENARIO")
        labels = sorted(
            [str(v) for v in chosen_df[chosen_col].dropna().unique() if re.match(r"^Α\d+$", str(v))],
            key=lambda x: int(re.search(r"\d+", x).group(0))
        )
        for lab in labels:
            sub = chosen_df.loc[chosen_df[chosen_col] == lab, ["ΟΝΟΜΑ", chosen_col]].copy()
            sub = sub.rename(columns={chosen_col: "ΤΜΗΜΑ"})
            sub.to_excel(w, index=False, sheet_name=str(lab))

    return {
        "chosen_sheet": chosen_sheet,
        "chosen_col": chosen_col,
        "broken_pairs": int(chosen_score["broken_friendships"]),
        "total_score": int(chosen_score["total_score"]),
        "diff_population": int(chosen_score["diff_population"]),
        "diff_gender_total": int(chosen_score["diff_gender_total"]),
        "diff_greek": int(chosen_score["diff_greek"]),
    }

if __name__ == "__main__":
    step1_6 = ROOT / "STEP1_6_PER_SCENARIO_20250912_160424.xlsx"
    final_out = ROOT / f"STEP7_FINAL_SCENARIO_MINRULE_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    info = pick_minrule_across_sheets(str(step1_6), str(final_out), seed=42)
    print({"file": final_out.name, **info})
