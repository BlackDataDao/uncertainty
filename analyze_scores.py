import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_rel
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json
from statsmodels.stats.contingency_tables import mcnemar

# Define roles and their short names
ROLES = {
    "stock broker": "Stock Broker",
    "stock broker whose role is to encourage clients to buy more crypto.": "Misleading Stock Broker",
    "risk manager": "Risk Manager",
    "risk manager whose role is to encourage clients to buy more crypto.": "Misleading Risk Manager",
}
ROLES = {
    "stock broker": "Stock Broker",
    "stock broker whose role is to encourage clients to buy more stocks.": "Misleading Stock Broker",
    "risk manager": "Risk Manager",
    "risk manager whose role is to encourage clients to buy more stocks.": "Misleading Risk Manager",
}
# ROLES = {
#     "stock broker": "Stock Broker",
#     "risk manager": "Risk Manager",
# }
def read_csv_with_encoding(file_path):
    """
    Try to read CSV file with different encodings.
    Returns DataFrame or None if all attempts fail.
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            print(f"Trying to read with {encoding} encoding...")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding}: {str(e)}")
            continue
    
    return None

def generate_statistics_excel(df, roles, output_folder, base_filename):
    """
    Generates an Excel file with statistics for each role's scores, role score pairs, and chi-square test results.
    """
    try:
        # Prepare data for Excel
        statistics_data = []
        chi_square_data = []

        # Significance level
        alpha = 0.05

        # Calculate mean scores for each role
        role_scores = {}
        for role in roles:
            col_name = f"{role}_score"
            if col_name in df.columns:
                role_scores[role] = pd.to_numeric(df[col_name], errors='coerce')
            else:
                print(f"⚠️ Warning: Could not find column for {role}")
                continue

        mean_scores = {role: scores.mean() for role, scores in role_scores.items()}
        for role, mean in mean_scores.items():
            short_role = ROLES.get(role, role)  # Get short name or use full name if not found
            statistics_data.append({"Role": short_role, "Mean Score": mean})

        # independent t-tests for role scores
        for i in range(len(roles)):
            if roles[i] not in role_scores:
                continue
            for j in range(i + 1, len(roles)):
                if roles[j] not in role_scores:
                    continue
                role1, role2 = roles[i], roles[j]
                short_role1 = ROLES.get(role1, role1)
                short_role2 = ROLES.get(role2, role2)
                t_stat, p_value = stats.ttest_ind(
                    role_scores[role1].dropna(),
                    role_scores[role2].dropna(),
                    nan_policy='omit'
                )
                mean_diff = mean_scores[role1] - mean_scores[role2]
                statistics_data.append({
                    "Role Pair": f"{short_role1} vs {short_role2}",
                    "Mean Difference": mean_diff,
                    "P-value (T-test)": p_value,
                    "Significant (T-test)": "Yes" if p_value < alpha else "No"
                })

        # paired t-tests for role scores
        for i in range(len(roles)):
            if roles[i] not in role_scores:
                continue
            for j in range(i + 1, len(roles)):
                if roles[j] not in role_scores:
                    continue
                role1, role2 = roles[i], roles[j]
                short_role1 = ROLES.get(role1, role1)
                short_role2 = ROLES.get(role2, role2)
                # 对齐索引，去除缺失值
                s1 = role_scores[role1]
                s2 = role_scores[role2]
                paired = pd.concat([s1, s2], axis=1).dropna()
                if paired.shape[0] == 0:
                    continue
                t_stat, p_value = ttest_rel(paired.iloc[:, 0], paired.iloc[:, 1])
                mean_diff = paired.iloc[:, 0].mean() - paired.iloc[:, 1].mean()
                statistics_data.append({
                    "Role Pair": f"{short_role1} vs {short_role2}",
                    "Mean Difference": mean_diff,
                    "P-value (Paired T-test)": p_value,
                    "Significant (Paired T-test)": "Yes" if p_value < alpha else "No"
                })
        print("\n=== Independent T-Tests and Paired T-Tests Results ===")
        for entry in statistics_data:
            if "Role Pair" in entry:
                print(f"{entry['Role Pair']}: Mean Difference = {entry['Mean Difference']:.3f}, "
                      f"P-value = {entry.get('P-value (T-test)', entry.get('P-value (Paired T-test)', 'N/A')):.3f}, "
                      f"Significant = {entry.get('Significant (T-test)', entry.get('Significant (Paired T-test)', 'N/A'))}")

        #  chi-square test for recommendation counts
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                role1, role2 = roles[i], roles[j]
                short_role1 = ROLES.get(role1, role1)
                short_role2 = ROLES.get(role2, role2)
                col_name1 = f"{role1}_recommendation"
                col_name2 = f"{role2}_recommendation"

                if col_name1 in df.columns and col_name2 in df.columns:
                    recommendations1 = df[col_name1].str.lower()
                    recommendations2 = df[col_name2].str.lower()

                    yes_count1 = (recommendations1 == 'yes').sum()
                    no_count1 = (recommendations1 == 'no').sum()
                    yes_count2 = (recommendations2 == 'yes').sum()
                    no_count2 = (recommendations2 == 'no').sum()

                    contingency_table = [[yes_count1, no_count1], [yes_count2, no_count2]]
                    chi2, p, dof, expected = chi2_contingency(contingency_table)

                    yes_diff = yes_count1 - yes_count2
                    chi_square_data.append({
                        "Role Pair": f"{short_role1} vs {short_role2}",
                        "Yes Count Difference": yes_diff,
                        "P-value (Chi-square)": p,
                        "Significant (Chi-square)": "Yes" if p < alpha else "No"
                    })
        print("\n=== Chi-Square Test Results ===")
        for entry in chi_square_data:
            if "Role Pair" in entry:
                print(f"{entry['Role Pair']}: Yes Count Difference = {entry['Yes Count Difference']}, "
                      f"P-value = {entry['P-value (Chi-square)']:.3f}, "
                      f"Significant = {entry['Significant (Chi-square)']}")

        # Create Excel file
        excel_file_path = os.path.join(output_folder, f"{base_filename}_statistics.xlsx")
        with pd.ExcelWriter(excel_file_path) as writer:
            pd.DataFrame(statistics_data).to_excel(writer, sheet_name="Role Scores Statistics", index=False)
            pd.DataFrame(chi_square_data).to_excel(writer, sheet_name="Chi-square Test Results", index=False)

        print(f"✅ Statistics Excel file saved as '{excel_file_path}'")
    except Exception as e:
        print(f"❌ Error generating statistics Excel file: {type(e).__name__}: {e}")

def test_recommendation_count_significance(df, roles):
    """
    Tests if the difference in recommendation counts ("Yes") between roles is statistically significant using a Chi-Square test.
    """
    try:
        # Create a contingency table
        observed = []
        for role in roles:
            col_name = f"{role}_recommendation"
            recommendations = df[col_name]
            recommendations = recommendations.str.lower()
            
            yes_count = (recommendations == 'yes').sum()
            no_count = (recommendations == 'no').sum()
            observed.append([yes_count, no_count])

        # Perform Chi-Square test
        chi2, p, dof, expected = chi2_contingency(observed)

        print("\n=== Chi-Square Test for Recommendation Significance ===")
        print(f"Chi-Square Statistic: {chi2:.3f}")
        print(f"P-value: {p:.3f}")
        print(f"Degrees of Freedom: {dof}")

        alpha = 0.05
        if p < alpha:
            print("The difference in recommendation counts between roles is statistically significant.")
        else:
            print("The difference in recommendation counts between roles is not statistically significant.")

    except Exception as e:
        print(f"❌ Error during Chi-Square test: {type(e).__name__}: {e}")

def test_recommendation_count_significance_mcnemar(df, roles):
    """
    使用McNemar检验对配对的Yes/No推荐数据进行显著性检验。
    """
    try:
        print("\n=== McNemar's Test for Paired Yes/No Recommendations ===")
        alpha = 0.05
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                role1, role2 = roles[i], roles[j]
                col1 = f"{role1}_recommendation"
                col2 = f"{role2}_recommendation"
                if col1 not in df.columns or col2 not in df.columns:
                    continue

                # 只保留两列都不缺失的数据
                paired = df[[col1, col2]].dropna()
                paired = paired.apply(lambda x: x.str.lower())

                # 构建2x2列联表
                #            role2_yes   role2_no
                # role1_yes   [a]         [b]
                # role1_no    [c]         [d]
                a = ((paired[col1] == 'yes') & (paired[col2] == 'yes')).sum()
                b = ((paired[col1] == 'yes') & (paired[col2] == 'no')).sum()
                c = ((paired[col1] == 'no') & (paired[col2] == 'yes')).sum()
                d = ((paired[col1] == 'no') & (paired[col2] == 'no')).sum()
                table = [[a, b], [c, d]]

                result = mcnemar(table, exact=True)
                short_role1 = ROLES.get(role1, role1)
                short_role2 = ROLES.get(role2, role2)
                print(f"\n{short_role1} vs {short_role2}:")
                print(f"Contingency Table: {table}")
                print(f"McNemar's test p-value: {result.pvalue:.4f}")
                print("Significant difference:" , "Yes" if result.pvalue < alpha else "No")
    except Exception as e:
        print(f"❌ Error during McNemar's test: {type(e).__name__}: {e}")

def save_mcnemar_results_to_excel(df, roles, output_folder, base_filename):
    """
    对每一对角色进行McNemar检验，并将结果保存到Excel文件中。
    """
    try:
        mcnemar_data = []
        alpha = 0.05
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                role1, role2 = roles[i], roles[j]
                col1 = f"{role1}_recommendation"
                col2 = f"{role2}_recommendation"
                if col1 not in df.columns or col2 not in df.columns:
                    continue

                paired = df[[col1, col2]].dropna()
                paired = paired.apply(lambda x: x.str.lower())

                a = ((paired[col1] == 'yes') & (paired[col2] == 'yes')).sum()
                b = ((paired[col1] == 'yes') & (paired[col2] == 'no')).sum()
                c = ((paired[col1] == 'no') & (paired[col2] == 'yes')).sum()
                d = ((paired[col1] == 'no') & (paired[col2] == 'no')).sum()
                table = [[a, b], [c, d]]

                result = mcnemar(table, exact=True)
                short_role1 = ROLES.get(role1, role1)
                short_role2 = ROLES.get(role2, role2)
                mcnemar_data.append({
                    "Role Pair": f"{short_role1} vs {short_role2}",
                    "b (yes→no)": b,
                    "c (no→yes)": c,
                    "McNemar p-value": result.pvalue,
                    "Significant (alpha=0.05)": "Yes" if result.pvalue < alpha else "No"
                })

        # 保存到Excel
        excel_file_path = os.path.join(output_folder, f"{base_filename}_mcnemar_results.xlsx")
        pd.DataFrame(mcnemar_data).to_excel(excel_file_path, index=False)
        print(f"✅ McNemar test results saved as '{excel_file_path}'")
    except Exception as e:
        print(f"❌ Error saving McNemar test results: {type(e).__name__}: {e}")

def analyze_role_scores(csv_file,roles=ROLES,output_folder_path=None):
    """
    Analyze role scores from the recommendations CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing role scores
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return

    # Try to read the CSV file with different encodings
    df = read_csv_with_encoding(csv_file)
    
    if df is None:
        print("❌ Error: Could not read the CSV file with any supported encoding")
        return
    
    # Create base filename and output folder
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    output_folder =output_folder_path+ f"analyze_{base_filename}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"✅ Output will be saved to folder: '{output_folder}'")
    
    # 1. Statistical Analysis
    print("\n=== Statistical Analysis of Role Scores ===")
    
    # Extract scores for each role
    roles = list(ROLES.keys())  # Use keys from ROLES dict
    role_scores = {}
    for role in roles:
        col_name = f"{role}_score"
        if col_name.replace(' ', ',') in df.columns:
            # Handle column names with commas instead of spaces
            role_scores[role] = pd.to_numeric(df[col_name.replace(' ', ',')], errors='coerce')
        elif col_name in df.columns:
            role_scores[role] = pd.to_numeric(df[col_name], errors='coerce')
        else:
            print(f"⚠️ Warning: Could not find column for {role}")
            continue
    
    # Calculate mean scores
    print("\nMean Scores:")
    mean_scores = {role: scores.mean() for role, scores in role_scores.items()}
    for role, mean in mean_scores.items():
        short_role = ROLES.get(role, role)
        print(f"{short_role}: {mean:.3f}")
    
    
    
    # Generate Excel file with statistics
    generate_statistics_excel(df, roles, output_folder, base_filename)
    
    # 2. Visualization - Group by Age and Role
    
    # Check if 'age' column exists
    if 'age' not in df.columns:
        print("⚠️ Warning: 'age' column not found, using default visualization for age-based plots")
        # age_groups = ["All"] # Not directly used in this structure anymore
        # df['age_group'] = "All" # Not directly used in this structure anymore
    # else:
        # Create age groups - this logic was specific to the old facet grid and might not be needed globally
        # age_bins = [0, 30, 40, 50, 60, 100]
        # age_labels = ['20-30', '31-40', '41-50', '51-60', '61+']
        # df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        # age_groups = df['age_group'].dropna().unique()

    generate_facet_grid_by_age(df, roles, base_filename, output_folder)
    generate_facet_grid_by_percentage(df, roles, base_filename, output_folder)
    generate_heatmap(df, roles, base_filename, output_folder)
    generate_interactive_plot(df, roles, base_filename, output_folder)
    generate_violin_plot(df, roles, base_filename, output_folder)
    generate_boxplot_matrix(df, roles, base_filename, output_folder)
    generate_recommendation_counts(df, roles, base_filename, output_folder)
    
    
    # 3. McNemar's test and save results to Excel
    save_mcnemar_results_to_excel(df, roles, output_folder, base_filename)

    # Save role short name mapping to JSON
    roles_mapping_path = os.path.join(output_folder, "roles_mapping.json")
    with open(roles_mapping_path, 'w') as f:
        json.dump(ROLES, f, indent=4)
    print(f"✅ Role short name mapping saved to '{roles_mapping_path}'")

    # Print role short name mapping
    print("\n=== Role Short Name Mapping ===")
    for role, short_role in ROLES.items():
        print(f"{short_role}: {role}")

def generate_facet_grid_by_age(df, roles, base_filename, output_folder):
    """
    Generates a facet grid with a chart for each age, showing investment scores by role and percentage,
    arranged in rows of five.
    """
    try:
        # Check if 'age' column exists
        if 'age' not in df.columns:
            print("⚠️ Warning: 'age' column not found, cannot generate facet grid by age.")
            return

        # Get unique ages
        ages = sorted(df['age'].dropna().unique())
        num_ages = len(ages)

        if num_ages == 0:
            print("⚠️ Warning: No unique age values found, cannot generate facet grid by age.")
            return

        # Calculate number of rows and columns for the grid
        num_cols = min(num_ages, 5)  # Up to 5 columns
        num_rows = (num_ages + num_cols - 1) // num_cols  # Calculate number of rows

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 6 * num_rows), sharex=True, sharey=True)
        fig.suptitle('Investment Scores by Role and Percentage (Faceted by Age)', fontsize=16)


        # Plot data for each age
        for i, age_val in enumerate(ages):
            row_index = i // num_cols
            col_index = i % num_cols

            # Handle cases where there's only one row or one column
            if num_rows == 1 and num_cols == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[col_index]
            elif num_cols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            ax.set_title(f'Age: {age_val}', fontsize=14)
            df_filtered_by_age = df[df['age'] == age_val]

            # Plot data for each role
            for role in roles:
                short_role = ROLES.get(role, role)
                # Get valid scores and percentages for this age and role
                scores = pd.to_numeric(df_filtered_by_age[f"{role}_score"], errors='coerce')
                percentages_for_plot = df_filtered_by_age['percentage']
                
                valid_indices = ~(scores.isna() | percentages_for_plot.isna())
                
                if valid_indices.sum() > 0:
                    plot_data = pd.DataFrame({
                        'percentage': percentages_for_plot[valid_indices], 
                        'score': scores[valid_indices]
                    }).sort_values('percentage')

                    if not plot_data.empty:
                        ax.plot(plot_data['percentage'], plot_data['score'],
                                label=short_role, marker='o', linestyle='-')

            # Set labels and legend
            ax.set_xlabel("Investment Percentage (%)", fontsize=12)
            ax.set_ylabel("Investment Score", fontsize=12)
            if ax.has_data():
                ax.legend(title="Role", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(num_ages, num_rows * num_cols):
            row_index = i // num_cols
            col_index = i % num_cols
            ax_to_remove = None
            if num_rows == 1 and num_cols == 1: pass
            elif num_rows == 1: ax_to_remove = axes[col_index] if num_cols > 1 else axes
            elif num_cols == 1: ax_to_remove = axes[row_index] if num_rows > 1 else axes
            else: ax_to_remove = axes[row_index, col_index]
            if ax_to_remove and hasattr(ax_to_remove, 'axis'): ax_to_remove.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_folder, f"facet_grid_by_age_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Facet grid by age saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating facet grid by age: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def generate_facet_grid_by_percentage(df, roles, base_filename, output_folder):
    """
    Generates a facet grid with a chart for each percentage value, 
    showing investment scores by role and age, arranged in rows of up to five.
    """
    try:
        # Check if necessary columns exist
        if 'percentage' not in df.columns or 'age' not in df.columns:
            print("⚠️ Warning: 'percentage' or 'age' column not found, cannot generate facet grid by percentage.")
            return

        # Get unique percentage values
        percentage_values = sorted(df['percentage'].dropna().unique())
        num_percentage_values = len(percentage_values)

        if num_percentage_values == 0:
            print("⚠️ Warning: No unique percentage values found, cannot generate facet grid by percentage.")
            return

        # Calculate number of rows and columns for the grid
        num_cols = min(num_percentage_values, 5)  # Up to 5 columns
        num_rows = (num_percentage_values + num_cols - 1) // num_cols  # Calculate number of rows

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 6 * num_rows), sharex=True, sharey=True)
        fig.suptitle('Investment Scores by Role and Age (Faceted by Investment Percentage)', fontsize=16)

        # Plot data for each percentage value
        for i, p_val in enumerate(percentage_values):
            row_index = i // num_cols
            col_index = i % num_cols

            # Handle cases where there's only one row or one column
            ax = None
            if num_rows == 1 and num_cols == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[col_index]
            elif num_cols == 1:
                ax = axes[row_index]
            else:
                ax = axes[row_index, col_index]

            ax.set_title(f'Percentage: {p_val}%', fontsize=14)

            # Filter data for the current percentage value
            df_filtered_by_percentage = df[df['percentage'] == p_val]

            # Plot data for each role
            for role in roles:
                short_role = ROLES.get(role, role)
                # Get valid scores and ages for this percentage and role
                scores = pd.to_numeric(df_filtered_by_percentage[f"{role}_score"], errors='coerce')
                ages_for_plot = df_filtered_by_percentage['age']
                
                valid_indices = ~(scores.isna() | ages_for_plot.isna())
                
                if valid_indices.sum() > 0:
                    plot_data = pd.DataFrame({
                        'age': ages_for_plot[valid_indices], 
                        'score': scores[valid_indices]
                    }).sort_values('age')

                    # Plot line if there's data
                    if not plot_data.empty:
                        ax.plot(plot_data['age'], plot_data['score'], label=short_role, marker='o', linestyle='-')

            # Set labels and legend
            ax.set_xlabel("Age", fontsize=12)
            ax.set_ylabel("Investment Score", fontsize=12)
            if ax.has_data(): # Only add legend if there is data plotted
                ax.legend(title="Role", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(num_percentage_values, num_rows * num_cols):
            row_index = i // num_cols
            col_index = i % num_cols
            ax_to_remove = None
            if num_rows == 1 and num_cols == 1: pass
            elif num_rows == 1: ax_to_remove = axes[col_index] if num_cols > 1 else axes
            elif num_cols == 1: ax_to_remove = axes[row_index] if num_rows > 1 else axes
            else: ax_to_remove = axes[row_index, col_index]
            if ax_to_remove and hasattr(ax_to_remove, 'axis'): ax_to_remove.axis('off')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        save_path = os.path.join(output_folder, f"facet_grid_by_percentage_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Facet grid by percentage saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating facet grid by percentage: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def generate_heatmap(df, roles, base_filename, output_folder):
    """
    Generates a heatmap of average investment scores by role and investment percentage range.
    """
    try:
        # Create pivot tables for average scores by percentage
        pivot_data = {}
        for role in roles:
            # Create a pivot table for each role
            pivot = pd.pivot_table(
                df, 
                values=f"{role}_score",
                index=df.index,  # Use index as a dummy index
                columns=pd.cut(df['percentage'], bins=[0,20,40,60,80,100]), 
                aggfunc='mean'
            )
            pivot_data[role] = pivot

        # Plot heatmaps
        fig, axes = plt.subplots(1, len(roles), figsize=(15, 5))
        for i, role in enumerate(roles):
            short_role = ROLES.get(role, role)
            sns.heatmap(pivot_data[role], annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i])
            axes[i].set_title(f"{short_role}")
            axes[i].set_ylabel("Data Point Index")  # Update y-axis label
            axes[i].set_xlabel("Investment % Range")

        plt.tight_layout()
        save_path = os.path.join(output_folder, f"heatmap_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Heatmap saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating heatmap: {type(e).__name__}: {e}")

def generate_interactive_plot(df, roles, base_filename, output_folder):
    """
    Generates an interactive scatter plot of investment scores by role and investment percentage using Plotly.
    """
    try:
        # Prepare data in long format
        plot_data = []
        for role in roles:
            scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            percentages = df['percentage']
            ages = df['age']
            
            for p, s, a in zip(percentages, scores, ages):
                if not pd.isna(s):
                    plot_data.append({
                        'Role': role,
                        'Percentage': p,
                        'Score': s,
                        'Age': a
                    })

        # Create interactive plot
        plot_df = pd.DataFrame(plot_data)
        fig = px.scatter(
            plot_df, 
            x='Percentage', 
            y='Score', 
            color='Role',
            hover_data=['Age', 'Role', 'Score'],
            category_orders={"Role": [ROLES.get(role, role) for role in roles]}  # Order roles by short name
        )

        fig.update_layout(legend_title="Role")  # Update legend title

        save_path = os.path.join(output_folder, f'interactive_{base_filename}.html')
        pio.write_html(fig, save_path)
        print(f"✅ Interactive plot saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating interactive plot: {type(e).__name__}: {e}")

def generate_violin_plot(df, roles, base_filename, output_folder):
    """
    Generates an interactive violin plot of investment scores by role using Plotly.
    """
    try:
        # Reshape data for Plotly
        plot_data = []
        for role in roles:
            role_scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            ages = df['age']
            for age, score in zip(ages, role_scores):
                if not pd.isna(score):
                    plot_data.append({
                        'Role': ROLES.get(role, role),  # Use short name
                        'Age': age,
                        'Score': score
                    })

        plot_df = pd.DataFrame(plot_data)

        # Create violin plot using Plotly
        fig = px.violin(
            plot_df, y="Score", color="Role", box=True, points="all",
            hover_data=['Age', 'Role', 'Score'],
            category_orders={"Role": [ROLES.get(role, role) for role in roles]}
        )

        fig.update_layout(
            title='Distribution of Scores by Role',
            yaxis_title='Score',
            legend_title="Role",
            legend=dict(
                orientation="h",           # 横向
                yanchor="bottom",
                y=-0.05,                   # 图例在图下方
                xanchor="center",
                x=0.5
            )
        )

        save_path = os.path.join(output_folder, f"violin_plot_{base_filename}.html")
        pio.write_html(fig, save_path)
        print(f"✅ Interactive violin plot saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating violin plot: {type(e).__name__}: {e}")

def generate_boxplot_matrix(df, roles, base_filename, output_folder):
    """
    Generates a matrix of box plots showing the distribution of investment scores by role.
    """
    try:
        # Create a grid of box plots
        fig, axes = plt.subplots(1, len(roles), figsize=(15, 6), sharey=True)
        for i, role in enumerate(roles):
            short_role = ROLES.get(role, role)
            scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            axes[i].boxplot(scores.dropna())
            axes[i].set_title(short_role)
            axes[i].set_ylabel('Score')

        plt.tight_layout()
        save_path = os.path.join(output_folder, f"boxplot_matrix_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Box plot matrix saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating box plot matrix: {type(e).__name__}: {e}")

def generate_recommendation_counts(df, roles, base_filename, output_folder):
    """
    Calculates and visualizes the number of Yes/No recommendations for each role.
    """
    try:
        # Calculate counts and percentages for each role
        counts_data = {}
        for role in roles: # roles here are full names
            col_name = f"{role}_recommendation"
            recommendations = df[col_name].str.lower() # Ensure lowercase for consistent matching
            yes_count = (recommendations == 'yes').sum()
            no_count = (recommendations == 'no').sum()
            total_count = yes_count + no_count
            
            yes_percent = (yes_count / total_count * 100) if total_count > 0 else 0
            no_percent = (no_count / total_count * 100) if total_count > 0 else 0
            
            counts_data[role] = { # keyed by full role name
                'Yes': yes_count, 
                'No': no_count,
                'Yes_Percent': yes_percent,
                'No_Percent': no_percent,
                'Total': total_count
            }

        # Create bar plot
        plt.figure(figsize=(14, 7)) # Increased figure size for better readability
        
        # Set up bars
        # roles_list contains short names, order matches counts_data.keys()
        roles_list = [ROLES.get(role, role) for role in counts_data.keys()]
        role_keys_for_plot = list(counts_data.keys()) # list of full role names, in order

        x = np.arange(len(roles_list))
        width = 0.35

        # Plot bars using percentages
        yes_percentages = [counts_data[key]['Yes_Percent'] for key in role_keys_for_plot]
        no_percentages = [counts_data[key]['No_Percent'] for key in role_keys_for_plot]

        yes_bars = plt.bar(x - width/2, yes_percentages, width, label='Yes', color='green')
        no_bars = plt.bar(x + width/2, no_percentages, width, label='No', color='red')

        # Customize plot
        plt.xlabel('Roles', fontsize=12)
        plt.ylabel('Percentage of Recommendations (%)', fontsize=12) 
        plt.title('Yes/No Recommendations by Role (Percentage)', fontsize=14) 
        plt.xticks(x, roles_list, rotation=45, ha="right") 
        plt.legend()
        plt.ylim(0, 100) 

        # Add value labels on the bars (displaying percentages)
        def autolabel(bars, full_role_keys, is_yes_bar):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                original_role_key = full_role_keys[i] # Get the full role name using the bar's index

                if is_yes_bar:
                    count_val = counts_data[original_role_key]['Yes']
                else:
                    count_val = counts_data[original_role_key]['No']

                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%\n(n={count_val})', # Display percentage and count
                        ha='center', va='bottom', fontsize=9)

        autolabel(yes_bars, role_keys_for_plot, True)
        autolabel(no_bars, role_keys_for_plot, False)

        # Save plot
        plt.tight_layout()
        save_path = os.path.join(output_folder, f"recommendation_counts_percentage_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        
        # Print counts and percentages
        print("\n=== Recommendation Counts and Percentages ===")
        for role_key in role_keys_for_plot: # Iterate in defined order
            short_role = ROLES.get(role_key, role_key)
            print(f"\n{short_role}:")
            print(f"Yes: {counts_data[role_key]['Yes']} ({counts_data[role_key]['Yes_Percent']:.1f}%)")
            print(f"No: {counts_data[role_key]['No']} ({counts_data[role_key]['No_Percent']:.1f}%)")
            print(f"Total: {counts_data[role_key]['Total']}")
        
        print(f"\n✅ Recommendation counts (percentage) plot saved as '{save_path}'")
        
    except Exception as e:
        print(f"❌ Error generating recommendation counts: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
# # 
# csv_file="r_gpt-4o-mini_recommendations_only_q2_20250615_161212_combine.csv"  # Replace with your actual CSV file path
# analyze_role_scores(csv_file="result/r_gpt-4o-mini_recommendations_only_q2_20250615_161212.csv", roles=ROLES, output_folder_path="result/")  # Replace with your actual CSV file path and output folder