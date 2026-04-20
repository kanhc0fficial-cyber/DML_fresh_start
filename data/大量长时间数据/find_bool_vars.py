import os
import glob
import time
from pathlib import Path
import pandas as pd
from python_calamine import CalamineWorkbook
import concurrent.futures
import csv
from multiprocessing import cpu_count

def is_bool_value(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return v == 0 or v == 1
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return True
        return v in ('0', '1', '0.0', '1.0', 'True', 'False')
    return False

def check_file(file_path):
    bool_vars = []
    try:
        wb = CalamineWorkbook.from_path(file_path)
        for sheet in wb.sheet_names:
            try:
                raw = wb.get_sheet_by_name(sheet).to_python()
                if not raw or len(raw) < 2:
                    continue
                
                header = [str(h).lower().strip() if h is not None else "" for h in raw[0]]
                if "value" not in header:
                    continue
                
                val_idx = header.index("value")
                is_bool = True
                has_value = False
                
                for row in raw[1:]:
                    if row is None or len(row) <= val_idx:
                        continue
                    v = row[val_idx]
                    if v is not None:
                        has_value = True
                        if not is_bool_value(v):
                            is_bool = False
                            break
                            
                if is_bool and has_value:
                    bool_vars.append(sheet)
            except Exception as e:
                pass
    except Exception as e:
        pass
    return file_path, bool_vars

def main():
    base_dir = Path(r"c:\backup\doubleml\大量长时间数据")
    files = list(base_dir.rglob("*.xlsx")) + list(base_dir.rglob("*.xls"))
    
    # We will limit workers to 4 to reduce memory pressure
    workers = min(4, cpu_count())
    print(f"Found {len(files)} files. Using {workers} cores.", flush=True)
    start_time = time.time()
    
    out_path = base_dir / "bool_variables_table.csv"
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Variable", "File"])
        
    processed_count = 0
    total_bools = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for fpath, bvars in executor.map(check_file, [str(f) for f in files]):
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(files)} files...", flush=True)
            
            if bvars:
                # append to csv immediately
                with open(out_path, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    for v in bvars:
                        writer.writerow([v, os.path.basename(fpath)])
                total_bools += len(bvars)
                
    elapsed = time.time() - start_time
    print(f"Scan complete in {elapsed:.2f} seconds.", flush=True)
    print(f"Found {total_bools} boolean variables.", flush=True)
    print(f"Results saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
