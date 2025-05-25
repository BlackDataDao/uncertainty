import csv
import itertools
import functools # Added import

# 1) Define your “axes” of variation here.
#    To add a new key later, just add it to this dict.
params = {
    "age":           list(range(25, 76, 5)),          # 25, 30, …, 75
    "net_cash":      list(range(5_000, 50_001, 5_000)),  # 5000, 10000, …, 50000
    "percentage":    list(range(5, 101, 5)),          # 5, 10, …, 100
    "product_type":  ["stock", "crypto"],
    "product_name":  ["google", "btc"],
    "gender":        ["male", "female"],
}

# 2) Prepare CSV output
output_file = "generated_profiles.csv"
fieldnames = list(params.keys())

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # 3) Loop over the cartesian product of all param values
    for combo in itertools.product(*params.values()):
        row = dict(zip(fieldnames, combo))
        writer.writerow(row)

# Calculate the number of records
num_records = functools.reduce(lambda a, b: a * b, [len(v) for v in params.values()])
print(f"Wrote {output_file} with {num_records} records.")
