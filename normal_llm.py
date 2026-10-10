import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from measurement import reading, complete_sum, coverage, stop_tracker, make_codecarbon_tracker, tracker_run_id
from question_order import BANK, load_bank, select_questions, identity, save_order
from model_selection import select_model
from cancellation import (cancellable_run, configure_run, begin_question,
                          track_phase, question_saved, query_local_model, protect_cleanup)

# Base paths
QUESTIONS_FILE = BANK

def query_llm(prompt: str, model: str) -> str:
    """Send prompt to local Ollama instance and return generated response."""
    return query_local_model(prompt, model)

def load_questions(path: Path = QUESTIONS_FILE):
    if Path(path).name == BANK.name:
        return load_bank(path)
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()
        
    df = pd.read_excel(path)
    
    # Normalize column names to avoid case-sensitivity errors
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if "questions" in df.columns:
        df = df.rename(columns={"questions": "Question"})
    elif "question" in df.columns:
        df = df.rename(columns={"question": "Question"})
    else:
        raise ValueError(f"'Question' column not found. Found: {df.columns.tolist()}")
        
    if "question id" in df.columns:
        df = df.rename(columns={"question id": "Question ID"})
    elif "Question ID" not in df.columns:
        df["Question ID"] = range(1, len(df) + 1)
        
    return df

def make_tracker(project_name: str, output_dir: Path, output_file: str = "emissions.csv"):
    return make_codecarbon_tracker('inference', project_name, output_dir, output_file)

def latest_row_for_run(emissions_csv: Path, known_run_ids: set, expected_run_id=None):
    """
    Read CodeCarbon tracking file and return the most recent row
    whose run_id is not in known_run_ids.
    """
    if not emissions_csv.exists():
        return None
    try:
        df = pd.read_csv(emissions_csv)
    except Exception:
        return None
    if df.empty:
        return None
        
    if "run_id" in df.columns:
        if expected_run_id is None:
            return None
        new_df = df[(df["run_id"].astype(str) == str(expected_run_id)) & ~df["run_id"].astype(str).isin(known_run_ids)]
        if not new_df.empty:
            return new_df.iloc[-1].to_dict()
            
    return None  # Never substitute an earlier session's measurements.

@cancellable_run
def process_queries(queries_df: pd.DataFrame, batch_mode: bool, model_name: str):
    # Generate a unique timestamp string for this specific run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Route to a permanent timestamped folder for batches, or a temp folder for manual tests
    if batch_mode:
        run_dir = Path(f"emissions_reports/exp1_llm_only_{timestamp}")
    else:
        run_dir = Path("emissions_reports/temp_manual_run")
        
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_run(run_dir, len(queries_df), model_name, rag=False)
    print(f"DEBUG TARGET FOLDER: {run_dir}")
    emissions_filename = "emissions.csv" if batch_mode else "temp_emissions.csv"
    emissions_csv = run_dir / emissions_filename
    
    known_run_ids = set()
    if emissions_csv.exists() and batch_mode:
        try:
            old = pd.read_csv(emissions_csv)
            if "run_id" in old.columns:
                known_run_ids = set(old["run_id"].astype(str).tolist())
        except Exception:
            pass
    elif emissions_csv.exists() and not batch_mode:
        try:
            os.remove(emissions_csv)
        except Exception:
            pass

# ---> NEW: Clear old live answers to prevent column mismatch crashes
    live_csv = run_dir / "answers.csv"
    if live_csv.exists() and not batch_mode:
        try:
            os.remove(live_csv)
        except Exception:
            pass

# ---> NEW WARMUP BLOCK GOES HERE <---
    if batch_mode:
        save_order(queries_df, run_dir)
    if batch_mode:
        print("\nWarming up LLM into VRAM...")
        try:
            query_llm("Warmup", model=model_name)
        except Exception:
            pass

    rows = []
    run_start = time.time()
    
    for _, row_data in queries_df.iterrows():
        qid = row_data.get("Question ID", "custom")
        q = str(row_data["Question"])
        begin_question(identity(row_data, len(rows) + 1), qid, q)
        
        print(f"\n[LLM] Running prompt: {q[:80]}...")
        tracker_name = f"LLM_only_q{qid}_{uuid.uuid4().hex[:8]}"
        tracker = make_tracker(tracker_name, run_dir, output_file=emissions_filename)
        
        t0 = time.time()
        track_phase(tracker, 'generation')
        tracker.start()
        
        try:
            ans = query_llm(q, model=model_name)
            status = "ok"
            error_msg = ""
        except Exception as e:
            ans = ""
            status = "error"
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg}")
            
        emissions_kg = stop_tracker(tracker, run_dir)
        latency_s = time.time() - t0
        
        row = latest_row_for_run(emissions_csv, known_run_ids, tracker_run_id(tracker))
        if row and "run_id" in row:
            known_run_ids.add(str(row["run_id"]))
            
        print(f"\n--- ANSWER --- \n{ans}\n--------------")
        print(f"Latency: {latency_s:.2f}s")
        print(f"Total emissions: {emissions_kg * 1000:.6g} g CO2eq (nan = unavailable)")
        
        if not batch_mode and row:
            print(f"Total Energy: {reading(row, 'energy_consumed'):.6g} kWh")
            for component in ('cpu', 'gpu', 'ram'):
                print(f"{component.upper()} Energy: {reading(row, component + '_energy'):.6g} kWh")
        
        rows.append({
            **identity(row_data, len(rows) + 1),
            "question_id": qid,
            "question": q,
            "answer": ans,
            "model_name": model_name,
            "latency_s": latency_s,
            "status": status,
            "error": error_msg,
            "emissions_kg": emissions_kg,
            "emissions_g": emissions_kg * 1000 if emissions_kg is not None else None,
            "energy_kwh": reading(row, "energy_consumed"),
            "cpu_energy_kwh": reading(row, "cpu_energy"),
            "gpu_energy_kwh": reading(row, "gpu_energy"),
            "ram_energy_kwh": reading(row, "ram_energy"),
        })
        # Save each result as it completes.
        live_csv = run_dir / "answers.csv"
        with protect_cleanup(defer_interrupt=True):
            pd.DataFrame([rows[-1]]).to_csv(live_csv, mode='a', header=not live_csv.exists(), index=False)
            question_saved()
        
    total_runtime_s = time.time() - run_start
    answers_df = pd.DataFrame(rows)
    
    if batch_mode:
        answers_path = run_dir / "answers.csv"
        summary_path = run_dir / "summary.csv"
        
        success_df = answers_df[answers_df["status"] == "ok"]
        total_emissions_kg = complete_sum(answers_df["emissions_kg"])
        total_energy_kwh = complete_sum(answers_df["energy_kwh"])
        
        summary_df = pd.DataFrame([{
            "model_name": model_name,
            "num_queries": len(answers_df),
            "successful_queries": int((answers_df["status"] == "ok").sum()),
            "failed_queries": int((answers_df["status"] == "error").sum()),
            "total_runtime_s": total_runtime_s,
            "avg_latency_s": answers_df["latency_s"].mean(),
            "median_latency_s": answers_df["latency_s"].median(),
            "total_emissions_kg": total_emissions_kg,
            "total_energy_kwh": total_energy_kwh,
            "avg_emissions_g_per_req": answers_df["emissions_g"].mean(),
            "median_emissions_g_per_req": answers_df["emissions_g"].median(),
        }])
        
        for metric in ("emissions_kg", "energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh"):
            for key, value in coverage(answers_df[metric]).items():
                summary_df[f"{metric}_{key}"] = value
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved batch results to {answers_path}")
        print(f"Saved batch summary to {summary_path}")
    else:
        if emissions_csv.exists():
            try:
                os.remove(emissions_csv)
            except Exception:
                pass


def main():
    while True:
        print("\n" + "="*45)
        print("Baseline LLM Query Options:")
        print("1. Run a specific question (choose topic sheet and serial number)")
        print("2. Run the entire question bank")
        print("3. Run a range (choose topic sheet and serial numbers)")
        print("4. Write a custom prompt")
        print("5. Exit")
        print("="*45)
        
        choice = input("Select an option (1-5): ").strip()
        if choice in ('1', '2', '3', '4'):
            try:
                model_name = select_model()
            except ValueError as exc:
                print(exc)
                continue
        
        if choice in ('1', '2', '3'):
            try:
                df_q = select_questions(choice)
            except ValueError as exc:
                print(exc)
                continue
            process_queries(df_q, batch_mode=(choice != '1'), model_name=model_name)
            
        elif choice == '4':
            user_query = input("\nEnter your custom prompt: ").strip()
            if not user_query:
                print("Prompt cannot be empty.")
                continue
            df_q = pd.DataFrame([{"Question ID": "custom", "Question": user_query}])
            process_queries(df_q, batch_mode=False, model_name=model_name)
            
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid selection. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nExited.')




