# src/register.py
from pathlib import Path

def main():
    run_id_path = Path("artifacts/run_id.txt")
    run_id = run_id_path.read_text(encoding="utf-8").strip()
    print(f"Register step: got run_id={run_id}")
    # Later: use MLflow client to register + transition stages.

if __name__ == "__main__":
    main()