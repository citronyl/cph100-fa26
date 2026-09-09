import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path



def expand_grid(grid):
	"""Return one dictionary for each combination in the parameter grid."""
	parameters = grid.get("parameters", grid)
	names = list(parameters)
	choices = [
	    values if isinstance(values, list) else [values]
		for values in (parameters[name] for name in names)
	]
	return [dict(zip(names, combination)) for combination in itertools.product(*choices)]

def grid_search_instance(item):
	"""Run a single instance of the grid search."""
	config, idx = item
	command = [sys.executable, "main.py"]

	for hyperparameter, value in config.items():
		command.extend([f"--{hyperparameter}", str(value)])

	temp_results_path = f"temp_results_{idx}.json"
	command.extend(["--results_path", temp_results_path])

	subprocess.run(command, check=True)

def aggregate_results(config, output_csv_path):
	"""Aggregate the results from all the temporary result files into a single CSV."""
	rows = []

	for idx, config in enumerate(config):
		temp_file = Path(f"temp_results_{idx}.json")
		train_auc, val_auc = None, None

		if temp_file.exists():
			with temp_file.open(encoding="utf-8") as f:
				data = json.load(f)
				train_auc = data.get("train_auc")
				val_auc = data.get("val_auc")

			temp_file.unlink()  # Remove the temporary file after reading

		rows.append({
			**config,
			"train_auc": train_auc,
			"val_auc": val_auc
        })

	rows.sort(key=lambda x: (x["val_auc"] is None, x["val_auc"]), reverse=True)

	if rows:
		with Path(output_csv_path).open("w", newline="", encoding="utf-8") as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
			writer.writeheader()
			writer.writerows(rows)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid", default="grid_search.json")
	parser.add_argument("--output", default="grid_search_results.csv")
	parser.add_argument("--max_workers", type=int, default=4)
	args = parser.parse_args()

	with Path(args.grid).open(encoding="utf-8") as file:
		grid = json.load(file)

	names = list(grid.keys())
	choices = [grid[name] if isinstance(grid[name], list) else [grid[name]] for name in names]
	configurations = expand_grid(grid)

	work_items = [(config, idx) for idx, config in enumerate(configurations)]

	with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
		list(executor.map(grid_search_instance, work_items))

	aggregate_results(configurations, args.output)


if __name__ == "__main__":
	main()
