import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class PredictionLogger:
    def __init__(self, storage_path: str = "./predictions"):
        self.path = Path(storage_path)
        self.path.mkdir(parents=True, exist_ok=True)

    def log_prediction(self, data: Dict[str, Any])):
        filename = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(self.path / filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved prediction to {filename}")

    def record_actual_result(self, MatchID: str, result: str)):
        pass
