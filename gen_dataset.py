import csv
import itertools
import functools

# 1) Define your “axes” of variation here.
#    To add a new key later, just add it to this dict.
params = {
    "age":           list(range(25, 71, 5)),          # 25, 30, …, 75
    "net_cash":     [5000, 10000, 50000, 100000, 500000, 1000000,5000000,10000000],  # 5000, 10000, …, 50000
    "percentage":    list(range(5, 101, 5)),          # 5, 10, …, 100
    "product_type":  ["stock", "bond", "crypto"],
}

# Define valid product names for each product type
product_type_to_names = {
    "stock": ["GOOGLE", "APPLE", "TESLA", "JPMorgan", "Coca-Cola"],
    "bond": ["US TREASURY 10Y", "US TREASURY 30Y", "APPLE CORPORATE BOND 5Y", "TESLA CORPORATE BOND 10Y","GOOGLE CORPORATE BOND 3Y"],
    "crypto": ["BTC", "ETH", "SOL", "XRP", "DOGE"],
}

# 2) Prepare CSV output

output_file = "generated_profiles_3.csv"
fieldnames = list(params.keys()) + ["product_name"]  # Add product_name to fieldnames

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # 3) Loop over the cartesian product of all param values
    for combo in itertools.product(*params.values()):
        row = dict(zip(fieldnames[:-1], combo))  # Exclude product_name for now
        product_type = row["product_type"]
        
        # Add valid product names for the respective product type
        for product_name in product_type_to_names.get(product_type, []):
            row["product_name"] = product_name
            writer.writerow(row)

# Calculate the number of records
num_records = functools.reduce(lambda a, b: a * b, [len(v) for k, v in params.items() if k != 'product_type']) * sum(len(v) for v in product_type_to_names.values())
print(f"Wrote {output_file} with {num_records} records.")
