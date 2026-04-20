import json
import os

class SavedStateManager():
    def __init__(self):
        self.RUNS_DIR = "runs"

    def ensure_runs_dir(self):
        if not os.path.exists(self.RUNS_DIR):
            os.makedirs(self.RUNS_DIR)


    def save(self, summary_results, detailed_results):
        self.ensure_runs_dir()
        run_id = summary_results["run_id"]
            
        filename = f"run_{run_id}.json"
        filepath = os.path.join(self.RUNS_DIR, filename)

        data = {
            "summary": summary_results,
            "details": detailed_results
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        return run_id, filepath
    

    def load(self,run_id):
        filename = f"run_{run_id}.json"
        filepath = os.path.join(self.RUNS_DIR, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No run found for ID {run_id}")

        with open(filepath, "r") as f:
            return json.load(f)
        

    def show_run_details(self,run_id):
        data = self.load(run_id)

        print("\nSummary:")
        for k, v in data["summary"].items():
            print(f"  {k}: {v}")

        print("\nDetails:")
        for k, v in data["details"].items():
            print(f"  {k}: {v}")

