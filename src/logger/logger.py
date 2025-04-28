import os
import json
from datetime import datetime
import time
import pandas as pd

class Logger:
    def __init__(self, save_dir="logs", run_name=None, args=None):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name if run_name is not None else f"run_{timestamp}"
        self.run_dir = os.path.join(save_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Save args if provided
        if args:
            args_dict = vars(args).copy()
            args_dict["timestamp"] = timestamp
            with open(os.path.join(self.run_dir, "args.json"), "w") as f:
                json.dump(args_dict, f, indent=4)

        # Initialize timing logs
        self.start_time = None
        self.end_time = None
        self.time_taken = None

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()
        self.time_taken = self.end_time - self.start_time

    def log_time(self):
        # Save the time information into a log file
        time_info = {
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "time_taken_seconds": self.time_taken,
        }
        with open(os.path.join(self.run_dir, "time_log.json"), "w") as f:
            json.dump(time_info, f, indent=4)

    def log_df(self, df: pd.DataFrame, filename):
        df.to_csv(os.path.join(self.run_dir, filename), index=True)

    def get_run_dir(self):
        return self.run_dir

