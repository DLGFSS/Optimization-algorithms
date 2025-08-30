import json

def extract_means(path):
    with open(path, 'r') as f:
        data = json.load(f)
    baseline = axo = None
    for b in data.get("benchmarks", []):
        group = b.get("group", "")
        mean = b.get("stats", {}).get("mean")
        if group.startswith("baseline") and mean:
            baseline = mean
        elif group.startswith("axo") and mean:
            axo = mean
    return baseline, axo

def calculate_overhead(baseline, axo):
    if baseline is None or axo is None:
        return None
    return ((axo - baseline) / baseline) * 100

def format_row(name, baseline, axo, overhead):
    return f"| {name:<24} | {baseline:>10.2f} | {axo:>8.2f} | {overhead:>11.2f}% |"

files = {
    "Local Search": "tests/benchmark/reports/local_search.json",
    "Simulated Annealing": "tests/benchmark/reports/simulated_annealing.json"
}

print("\n Overhead Axo y Baseline\n")
print("| Algoritmo               | Baseline (µs) | Axo (µs) | Overhead (%) |")
print("|-------------------------|---------------|----------|--------------|")

for name, path in files.items():
    baseline, axo = extract_means(path)
    if baseline is not None and axo is not None:
        print(format_row(name, baseline * 1e6, axo * 1e6, calculate_overhead(baseline, axo)))
    else:
        print(f"| {name:<24} |     error     |   error  |     error    |")
