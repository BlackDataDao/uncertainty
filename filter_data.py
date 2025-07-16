"""
Module: filter_data.py

Provides utility functions to filter and sample the synthetic investment dataset (SIDD.csv).

Functions:
  - filter_data_by_input():
      Filters the dataset by predefined criteria (age, product_type, product_name,
      percentage, net_cash), saves the result to "filtered_profiles_filtered_2.csv",
      and returns the filtered DataFrame.
  - filter_random_data_by_product_type(n, product_types, output_folder):
      Filters by given product types and other criteria, selects n random rows,
      saves them to a timestamped CSV in the specified folder, and returns the file path.

Dependencies:
  pandas, random

Usage:
    from filter_data import filter_data_by_input, filter_random_data_by_product_type
    df = filter_data_by_input()
    random_file = filter_random_data_by_product_type(5, ["crypto", "stock"], "groups/group1/")
"""
import pandas as pd
import random

DATA_SET_FILE = "SIDD.csv"

def filter_data_by_input():
    """
    Filters the dataset based on specific criteria and saves the result to a new CSV file.
    """

    # 1) Load the full dataset
    df = pd.read_csv(DATA_SET_FILE)

    # 2) Build a mask of your filtering conditions
    mask = (
        df["age"].between(20, 70) &
        (df["product_type"] == "stock") &
        (df["product_name"] == "google") &
        df["percentage"].between(1, 100) &
        (df["net_cash"].between(8000, 10000)) 
    )

    # 3) Apply and save or inspect
    filtered_df = df[mask]
    filtered_df.to_csv("filtered_profiles_filtered_2.csv", index=False)

    print(f"{len(filtered_df)} rows matched your criteria.")

    return filtered_df

def filter_random_data_by_product_type(n, product_types,output_folder):
    """
    Filters the dataset based on specific product types and selects n random rows.
    
    Args:
        n (int): Number of random rows to select.
        product_types (list): List of product types to filter (e.g., ["stock", "crypto"]).
    
    Returns:
        pd.DataFrame: A DataFrame containing n random rows matching the criteria.
    """
    # 1) Load the full dataset
    df = pd.read_csv(DATA_SET_FILE)

    # 2) Build a mask for filtering by product types
    mask = (
        df["product_type"].isin(product_types)&
        df["age"].between(20, 70) &  # Example age filter
        df["percentage"].between(0, 100)   # Example percentage filter
    )
    # 3) Apply the mask to filter the dataset
    filtered_df = df[mask]

    # 4) Select n random rows from the filtered dataset
    if len(filtered_df) < n:
        print(f"⚠️ Warning: Requested {n} rows, but only {len(filtered_df)} rows match the criteria.")
        n = len(filtered_df)  # Adjust n to the available rows

    random_rows = filtered_df.sample(n=n, random_state=random.randint(0, len(filtered_df) - 1))

    # 5) Generate the output file name dynamically based on product types
    product_types_str = "_".join(product_types)
    timestamp = pd.Timestamp.now().strftime("%m%d%H%M")
    output_file = output_folder+f"data_random_{n}_{product_types_str}_{timestamp}.csv"

    # 6) Save the filtered random rows to the output file
    random_rows.to_csv(output_file, index=False)

    print(f"{len(random_rows)} random rows saved to '{output_file}'.")
    return output_file

# Example usage
# filter_random_data_by_product_type(200, ["crypto", "stock"], "groups/group3/")

