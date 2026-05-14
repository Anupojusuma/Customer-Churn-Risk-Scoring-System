from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import save_prediction_report, score_customers
from src.utils import DATA_PATH, DEFAULT_MODEL_PATH, REPORTS_DIR, read_customer_data, resolve_model_path


def main() -> None:
    df = read_customer_data(DATA_PATH)
    scored = score_customers(df, model_path=resolve_model_path(DEFAULT_MODEL_PATH))
    output_path = REPORTS_DIR / "churn_scored_customers.csv"
    scored.to_csv(output_path, index=False)
    save_prediction_report(scored, REPORTS_DIR / "prediction_report.csv")
    print(f"Batch scoring completed. Saved scored data to {output_path}")


if __name__ == "__main__":
    main()
