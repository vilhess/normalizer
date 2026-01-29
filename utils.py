import json

def save_results_json(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f)