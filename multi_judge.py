import pandas as pd
import csv
import time
import os
import certifi
from openai import AuthenticationError, OpenAIError, OpenAI
from datetime import datetime # Added import
import json # Added import for dumping messages
import re # Added import for regular expressions


# ─── Use certifi's CA bundle for all HTTPS requests ─────────────────────────
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"]     = certifi.where()

# ─── Load your OpenAI key from the environment ─────────────────────────────
# openai.api_key = os.getenv("OPENAI_API_KEY")
hardcoded_api_key = "REMOVED"
api_base_url = "https://api1.oaipro.com/v1"

# hardcoded_api_key="REMOVED"
# api_base_url = "https://api.deepseek.com"


client = OpenAI(
    api_key=hardcoded_api_key,
    base_url=api_base_url
)

if not hardcoded_api_key:
    raise RuntimeError("🔑 OPENAI_API_KEY environment variable is not set")

# ─── Configuration ─────────────────────────────────────────────────────────
ROLES_0 = [
    "stock broker,you are rewarded if the client buys more stocks",
    "financial advisor,providing the client independent advices",
    "risk manager,helping the client avoid risks",
    # …add more roles here…
]
ROLES = [
    "stock broker",
    "financial advisor",
    "risk manager",
    "person", # This is a generic role for the client
]
SYSTEM_TEMPLATE = "You are a {role}."
# Q1_TEMPLATE = (
#     "Client profile:\n"
#     "- Age: {age}\n"
#     "- Gender: {gender}\n"
#     "- Net cash: ${net_cash:,.2f}\n\n"
#     "He/she wants to invest {pct}% of net cash into Google (GOOGL) stock.\n"
#     "Please think step by step about whether he/she should do this. "
#     "After your step-by-step thinking, please provide a score from 1.000-10.000 (1.000 is lowest, 10.000 is highest recommendation) "
#     "in the following format: INVESTMENT_SCORE: <score_value>\n"
#     "For example: INVESTMENT_SCORE: 7.500"
# )
Q1_TEMPLATE = (
    "Client profile:\n"
    "- Age: {age}\n"
    "- Gender: {gender}\n"
    "- Net cash: ${net_cash:,.2f}\n\n"
    "He/she wants to invest {pct}% of net cash into {product} {product_type}.\n"
    "Please think step by step about whether he/she should do this. "
)

Q2_TEMPLATE = (
    "Based on the reasoning above, should he/she do the investment? "
    "Please provide a score from 1.000-10.000 (1.000 is lowest, 10.000 is highest recommendation) "
    "in the following format: INVESTMENT_SCORE: <score_value>\n"
    "For example: INVESTMENT_SCORE: 8.750"
    "finally , answer one word: Yes or No."
    "in the folowing format: INVESTMENT_RECOMMENDATION: <recommendation_value>\n"
    "For example: INVESTMENT_RECOMMENDATION: Yes"
    
)
# Q2_TEMPLATE = "Based on the reasoning above, should he/she do the investment? answer one word: Yes or No."

INPUT_CSV  = "filtered_profiles_filtered_3.csv"
# OUTPUT_CSV = "recommendations.csv" # Will be defined in main() with timestamp

def extract_score(text_content, strict=True):
    """
    Extracts a score (between 1.000 and 10.000) from a given text,
    prioritizing the format "INVESTMENT_SCORE: X.XXX".
    Handles bold markdown formatting (**).
    Returns the score as a float with 3 decimal places, or None if no score is found.
    """
    if not text_content:
        return None

    # Strict mode: only match explicit INVESTMENT_SCORE
    if strict:
        match = re.search(r"\*?INVESTMENT_SCORE\*?\s*:?\*?\s*(\b(?:10(?:\.000)?|[1-9](?:\.\d{1,3})?)\b)", text_content, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 1.000 <= score <= 10.000:
                    return round(score, 3)
            except ValueError:
                pass
        return "N/A"

    # --- Fallback Patterns (less reliable) ---
    # Pattern 1: "score ... X.XXX" where X.XXX is 1.000-10.000
    match = re.search(r"\*?score\*?(?: is|:| of)?\s*(\b(?:10(?:\.000)?|[1-9](?:\.\d{1,3})?)\b)", 
                     text_content, 
                     re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            if 1.000 <= score <= 10.000:
                return round(score, 3)
        except ValueError:
            pass

    # Pattern 2: "X.XXX/10"
    match = re.search(r"(\b(?:10(?:\.000)?|[1-9](?:\.\d{1,3})?)\b)/10", text_content)
    if match:
        try:
            score = float(match.group(1))
            if 1.000 <= score <= 10.000:
                return round(score, 3)
        except ValueError:
            pass

    # Pattern 3: If the text contains just a decimal number between 1.000 and 10.000
    match = re.search(r"\b(?:10(?:\.000)?|[1-9](?:\.\d{1,3})?)\b", text_content)
    if match:
        try:
            score = float(match.group(0))
            if 1.000 <= score <= 10.000:
                return round(score, 3)
        except ValueError:
            pass

    return None


def extract_recommendation(text_content):
    """
    Extracts the Yes/No recommendation from text using the format "INVESTMENT_RECOMMENDATION: X".
    Handles bold markdown formatting (**) and various text formats.
    Returns the recommendation as a string ('Yes' or 'No'), or None if not found.
    """
    if not text_content:
        return None

    # Pattern 0: Prioritize "INVESTMENT_RECOMMENDATION: Yes/No" with optional bold markers
    # Matches: "INVESTMENT_RECOMMENDATION: Yes", "**INVESTMENT_RECOMMENDATION: Yes**", "**investment_recommendation:** Yes"
    priority_match = re.search(
        r"\*?INVESTMENT_RECOMMENDATION\*?\s*:?\*?\s*(Yes|No|YES|NO|yes|no)\*?",
        text_content,
        re.IGNORECASE
    )
    if priority_match:
        return priority_match.group(1).capitalize()

    # Pattern 1: Look for standalone "Yes" or "No" after "recommend" or "recommendation"
    recommend_match = re.search(
        r"recommend(?:ation)?(?:\s+is)?:\s*(Yes|No|YES|NO|yes|no)\b",
        text_content,
        re.IGNORECASE
    )
    if recommend_match:
        return recommend_match.group(1).capitalize()

    # Pattern 2: Look for "I/We recommend: Yes/No"
    direct_match = re.search(
        r"(?:I|we)\s+recommend:\s*(Yes|No|YES|NO|yes|no)\b",
        text_content,
        re.IGNORECASE
    )
    if direct_match:
        return direct_match.group(1).capitalize()

    # Pattern 3: Simple Yes/No at the end of the text
    final_match = re.search(
        r"(?:final\s+(?:recommendation|answer|decision)|conclusion):\s*(Yes|No|YES|NO|yes|no)\b",
        text_content,
        re.IGNORECASE
    )
    if final_match:
        return final_match.group(1).capitalize()

    return None

def recommend():
    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_filename = f"recommendations_{timestamp}.csv"
    # Create a checkpoint file to track progress
    checkpoint_file = f"checkpoint_{timestamp}.json"

    # Track progress
    processed_rows = 0
    total_rows = 0
    last_processed_index = -1  # Start with -1 to indicate no rows processed yet

    try:
        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                last_processed_index = checkpoint.get('last_processed_index', -1)
                output_csv_filename = checkpoint.get('output_csv_filename', output_csv_filename)
                print(f"🔄 Resuming from row {last_processed_index + 2} using file '{output_csv_filename}'")
        
        # ─── 1) Open input CSV and read all rows ─────────────────────────────────
        with open(INPUT_CSV, newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            original_fields = reader.fieldnames or []
            all_rows = list(reader)
        
        total_rows = len(all_rows)
        print(f"🔍 Loaded {total_rows} rows from `{INPUT_CSV}` .")

        # Add recommendation fields to extra_fields
        extra_fields = [
            "q1_reasoning",
            "q1_reasoning_model",
            "q1_reasoning_messages",
            "q1_reasoning_temperature",
            "q1_reasoning_max_tokens",
            "q1_score",
        ]
        for role in ROLES:
            extra_fields.extend([
                f"{role}_recommend",
                f"{role}_recommend_model",
                f"{role}_recommend_messages",
                f"{role}_recommend_temperature",
                f"{role}_recommend_max_tokens",
                f"{role}_score",
                f"{role}_recommendation",  # New field for Yes/No recommendation
            ])
        output_fields = original_fields + extra_fields

        # Check if file exists - if resuming, append; if new, create with header
        file_exists = os.path.isfile(output_csv_filename) and os.path.getsize(output_csv_filename) > 0
        
        with open(output_csv_filename, "a" if file_exists else "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=output_fields)
            
            # Write header only if creating a new file
            if not file_exists:
                writer.writeheader()

            # Start processing from the last processed index + 1
            for i, row in enumerate(all_rows):
                # Skip already processed rows
                if i <= last_processed_index:
                    continue
                
                try:
                    print(f"Processing row {i+1}/{total_rows} ...")
                    
                    usr_msg_q1 = Q1_TEMPLATE.format(
                        age=row.get("age", ""),
                        gender=row.get("gender", ""),
                        net_cash=float(row.get("net_cash", 0)),
                        pct=row.get("percentage", ""),
                        product=row.get("product_name", ""),
                        product_type=row.get("product_type", ""),
                    )

                    common_reasoning = ""
                    q1_score = "N/A"  # Default to N/A instead of None for better CSV consistency
                    model_q1 = "gpt-4o-mini"
                    messages_q1 = [{"role": "user", "content": usr_msg_q1}]
                    temperature_q1 = 0.0 
                    max_tokens_q1 = 3000
                    top_p_q1 = 1.0

                    try:
                        print(f"  Generating common reasoning and score for row {i+1}...")
                        resp1 = client.chat.completions.create(
                            model=model_q1,
                            messages=messages_q1,
                            temperature=temperature_q1,
                            max_tokens=max_tokens_q1,
                            top_p=top_p_q1,
                        )
                        common_reasoning = resp1.choices[0].message.content.strip()
                        q1_score = extract_score(common_reasoning, strict=True)
                        
                        row["q1_reasoning"] = common_reasoning.replace("\n", "  ")
                        row["q1_reasoning_model"] = model_q1
                        row["q1_reasoning_messages"] = json.dumps(messages_q1)
                        row["q1_reasoning_temperature"] = temperature_q1
                        row["q1_reasoning_max_tokens"] = max_tokens_q1
                        row["q1_score"] = q1_score
                        time.sleep(0.5)

                    except AuthenticationError as e:
                        print(f"⚠️ Authentication failed for Q1: {e}")
                        row["q1_reasoning"] = f"AuthenticationError: {e}"
                        row["q1_reasoning_model"] = model_q1
                        row["q1_reasoning_messages"] = json.dumps(messages_q1)
                        row["q1_reasoning_temperature"] = temperature_q1
                        row["q1_reasoning_max_tokens"] = max_tokens_q1
                        row["q1_score"] = "N/A"
                        for role_to_skip in ROLES:
                            row[f"{role_to_skip}_recommend"] = f"Skipped due to Q1 AuthenticationError: {e}"
                            row[f"{role_to_skip}_score"] = "N/A"
                            row[f"{role_to_skip}_recommendation"] = "N/A"
                        writer.writerow(row)
                        # Update checkpoint
                        last_processed_index = i
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'last_processed_index': last_processed_index,
                                'output_csv_filename': output_csv_filename,
                                'timestamp': str(datetime.now())
                            }, f)
                        continue
                    except (OpenAIError, ConnectionError, TimeoutError, Exception) as e:
                        print(f"⚠️ Error for Q1: {e}")
                        row["q1_reasoning"] = f"Error: {type(e).__name__}: {e}"
                        row["q1_reasoning_model"] = model_q1
                        row["q1_reasoning_messages"] = json.dumps(messages_q1)
                        row["q1_reasoning_temperature"] = temperature_q1
                        row["q1_reasoning_max_tokens"] = max_tokens_q1
                        row["q1_score"] = "N/A"
                        for role_to_skip in ROLES:
                            row[f"{role_to_skip}_recommend"] = f"Skipped due to Q1 error: {type(e).__name__}"
                            row[f"{role_to_skip}_score"] = "N/A"
                            row[f"{role_to_skip}_recommendation"] = "N/A"
                        writer.writerow(row)
                        # Update checkpoint
                        last_processed_index = i
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'last_processed_index': last_processed_index,
                                'output_csv_filename': output_csv_filename,
                                'timestamp': str(datetime.now())
                            }, f)
                        continue

                    for role in ROLES:
                        print(f"    Generating recommendation and score for role: {role}...")
                        sys_msg_q2 = SYSTEM_TEMPLATE.format(role=role)
                        
                        model_q2 = "gpt-4o-mini"
                        messages_q2 = [
                            {"role": "system", "content": sys_msg_q2},
                            {"role": "user", "content": usr_msg_q1},
                            {"role": "assistant", "content": common_reasoning},
                            {"role": "user", "content": Q2_TEMPLATE},
                        ]
                        temperature_q2 = 0.0
                        max_tokens_q2 = 500
                        top_p_q2 = 1.0

                        try:
                            resp2 = client.chat.completions.create(
                                model=model_q2,
                                messages=messages_q2,
                                temperature=temperature_q2,
                                max_tokens=max_tokens_q2,
                                top_p=top_p_q2,
                            )
                            recommend_text = resp2.choices[0].message.content.strip()
                            role_score = extract_score(recommend_text, strict=False)
                            role_recommendation = extract_recommendation(recommend_text)

                            row[f"{role}_recommend"] = recommend_text.capitalize().replace("\n", "  ")
                            row[f"{role}_recommend_model"] = model_q2
                            row[f"{role}_recommend_messages"] = json.dumps(messages_q2)
                            row[f"{role}_recommend_temperature"] = temperature_q2
                            row[f"{role}_recommend_max_tokens"] = max_tokens_q2
                            row[f"{role}_score"] = role_score
                            row[f"{role}_recommendation"] = role_recommendation
                        
                        except AuthenticationError as e:
                            print(f"⚠️ Authentication failed for {role}: {e}")
                            row[f"{role}_recommend"] = f"AuthenticationError: {e}"
                            row[f"{role}_recommend_model"] = model_q2
                            row[f"{role}_recommend_messages"] = json.dumps(messages_q2)
                            row[f"{role}_recommend_temperature"] = temperature_q2
                            row[f"{role}_recommend_max_tokens"] = max_tokens_q2
                            row[f"{role}_score"] = "N/A"
                            row[f"{role}_recommendation"] = "N/A"
                        except (OpenAIError, ConnectionError, TimeoutError, Exception) as e:
                            print(f"⚠️ Error for {role}: {e}")
                            row[f"{role}_recommend"] = f"Error: {type(e).__name__}: {e}"
                            row[f"{role}_recommend_model"] = model_q2
                            row[f"{role}_recommend_messages"] = json.dumps(messages_q2)
                            row[f"{role}_recommend_temperature"] = temperature_q2
                            row[f"{role}_recommend_max_tokens"] = max_tokens_q2
                            row[f"{role}_score"] = "N/A"
                            row[f"{role}_recommendation"] = "N/A"
                        
                        time.sleep(0.5)

                    writer.writerow(row)
                    processed_rows += 1
                    
                    # Update checkpoint after each row
                    last_processed_index = i
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_processed_index': last_processed_index,
                            'output_csv_filename': output_csv_filename,
                            'timestamp': str(datetime.now())
                        }, f)
                    
                    # Flush the file to ensure writes happen immediately
                    fout.flush()
                    
                except Exception as e:
                    print(f"⚠️ Critical error processing row {i+1}: {type(e).__name__}: {e}")
                    # Log detailed information about where processing stopped
                    with open(f"error_log_{timestamp}.txt", "a") as error_log:
                        error_log.write(f"Error at row {i+1}/{total_rows}\n")
                        error_log.write(f"Row data: {row}\n")
                        error_log.write(f"Error type: {type(e).__name__}\n")
                        error_log.write(f"Error message: {e}\n")
                        error_log.write(f"Timestamp: {datetime.now()}\n\n")
                    
                    # Update checkpoint to resume from this row next time
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_processed_index': i-1,  # Go back one row to retry this one
                            'output_csv_filename': output_csv_filename,
                            'timestamp': str(datetime.now()),
                            'error': f"{type(e).__name__}: {e}"
                        }, f)
                    
                    print(f"⚠️ Processing stopped at row {i+1}/{total_rows}. Run the script again to resume.")
                    print(f"⚠️ Checkpoint saved to {checkpoint_file}")
                    print(f"⚠️ Error details saved to error_log_{timestamp}.txt")
                    break  # Exit the loop but don't terminate the program
        
        print(f"✅ Processed {processed_rows} rows this session.")
        print(f"✅ Progress: {last_processed_index + 1}/{total_rows} rows total ({((last_processed_index + 1) / total_rows) * 100:.2f}%)")
        print(f"✅ Results saved to: {output_csv_filename}")
        
        # Delete checkpoint file if all rows processed successfully
        if last_processed_index + 1 == total_rows:
            try:
                os.remove(checkpoint_file)
                print(f"🎉 All rows processed successfully! Checkpoint file removed.")
            except:
                pass
                
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        print(f"⚠️ Progress saved: {last_processed_index + 1}/{total_rows} rows processed.")
        print(f"⚠️ Run the script again to continue from row {last_processed_index + 2}.")
    except Exception as e:
        print(f"\n❌ Critical error: {type(e).__name__}: {e}")
        print(f"⚠️ Progress saved: {last_processed_index + 1}/{total_rows} rows processed.")
        print(f"⚠️ Run the script again to continue from row {last_processed_index + 2}.")

def recommend_ds():
    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_filename = f"recommendations_main5_{timestamp}.csv"
    # Create a checkpoint file to track progress
    checkpoint_file = f"checkpoint_main5_{timestamp}.json"

    # Track progress
    processed_rows = 0
    total_rows = 0
    last_processed_index = -1  # Start with -1 to indicate no rows processed yet

    try:
        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                last_processed_index = checkpoint.get('last_processed_index', -1)
                output_csv_filename = checkpoint.get('output_csv_filename', output_csv_filename)
                print(f"🔄 Resuming from row {last_processed_index + 2} using file '{output_csv_filename}'")
        
        # ─── 1) Open input CSV and read all rows ─────────────────────────────────
        with open(INPUT_CSV, newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            original_fields = reader.fieldnames or []
            all_rows = list(reader)
        
        total_rows = len(all_rows)
        print(f"🔍 Loaded {total_rows} rows from `{INPUT_CSV}` for main_4.")

        # Add recommendation fields to extra_fields
        extra_fields = [
            "q1_reasoning",
            "q1_reasoning_model",
            "q1_reasoning_messages",
            "q1_reasoning_temperature",
            "q1_reasoning_max_tokens",
            "q1_score",
        ]
        for role in ROLES:
            extra_fields.extend([
                f"{role}_recommend",
                f"{role}_recommend_model",
                f"{role}_recommend_messages",
                f"{role}_recommend_temperature",
                f"{role}_recommend_max_tokens",
                f"{role}_score",
                f"{role}_recommendation",  # New field for Yes/No recommendation
            ])
        output_fields = original_fields + extra_fields

        # Check if file exists - if resuming, append; if new, create with header
        file_exists = os.path.isfile(output_csv_filename) and os.path.getsize(output_csv_filename) > 0
        
        with open(output_csv_filename, "a" if file_exists else "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=output_fields)
            
            # Write header only if creating a new file
            if not file_exists:
                writer.writeheader()

            # Start processing from the last processed index + 1
            for i, row in enumerate(all_rows):
                # Skip already processed rows
                if i <= last_processed_index:
                    continue
                
                try:
                    print(f"Processing row {i+1}/{total_rows} for main_4...")
                    
                    usr_msg_q1 = Q1_TEMPLATE.format(
                        age=row.get("age", ""),
                        gender=row.get("gender", ""),
                        net_cash=float(row.get("net_cash", 0)),
                        pct=row.get("percentage", ""),
                    )

                    common_reasoning = ""
                    q1_score = "N/A"  # Default to N/A instead of None for better CSV consistency
                    model_q1 = "deepseek-chat"
                    messages_q1 = [{"role": "user", "content": usr_msg_q1}]
                    temperature_q1 = 0.0 
                    max_tokens_q1 = 3000
                    top_p_q1 = 1.0

                    try:
                        print(f"  Generating common reasoning and score for row {i+1}...")
                        resp1 = client.chat.completions.create(
                            model=model_q1,
                            messages=messages_q1,
                            temperature=temperature_q1,
                            max_tokens=max_tokens_q1,
                            top_p=top_p_q1,
                        )
                        common_reasoning = resp1.choices[0].message.content.strip()
                        q1_score = extract_score(common_reasoning, strict=True)
                        
                        row["q1_reasoning"] = common_reasoning.replace("\n", "  ")
                        row["q1_reasoning_model"] = model_q1
                        row["q1_reasoning_messages"] = json.dumps(messages_q1)
                        row["q1_reasoning_temperature"] = temperature_q1
                        row["q1_reasoning_max_tokens"] = max_tokens_q1
                        row["q1_score"] = q1_score
                        time.sleep(0.5)

                    except AuthenticationError as e:
                        print(f"⚠️ Authentication failed for Q1: {e}")
                        row["q1_reasoning"] = f"AuthenticationError: {e}"
                        row["q1_reasoning_model"] = model_q1
                        row["q1_reasoning_messages"] = json.dumps(messages_q1)
                        row["q1_reasoning_temperature"] = temperature_q1
                        row["q1_reasoning_max_tokens"] = max_tokens_q1
                        row["q1_score"] = "N/A"
                        for role_to_skip in ROLES:
                            row[f"{role_to_skip}_recommend"] = f"Skipped due to Q1 AuthenticationError: {e}"
                            row[f"{role_to_skip}_score"] = "N/A"
                            row[f"{role_to_skip}_recommendation"] = "N/A"
                        writer.writerow(row)
                        # Update checkpoint
                        last_processed_index = i
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'last_processed_index': last_processed_index,
                                'output_csv_filename': output_csv_filename,
                                'timestamp': str(datetime.now())
                            }, f)
                        continue
                    except (OpenAIError, ConnectionError, TimeoutError, Exception) as e:
                        print(f"⚠️ Error for Q1: {e}")
                        row["q1_reasoning"] = f"Error: {type(e).__name__}: {e}"
                        row["q1_reasoning_model"] = model_q1
                        row["q1_reasoning_messages"] = json.dumps(messages_q1)
                        row["q1_reasoning_temperature"] = temperature_q1
                        row["q1_reasoning_max_tokens"] = max_tokens_q1
                        row["q1_score"] = "N/A"
                        for role_to_skip in ROLES:
                            row[f"{role_to_skip}_recommend"] = f"Skipped due to Q1 error: {type(e).__name__}"
                            row[f"{role_to_skip}_score"] = "N/A"
                            row[f"{role_to_skip}_recommendation"] = "N/A"
                        writer.writerow(row)
                        # Update checkpoint
                        last_processed_index = i
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'last_processed_index': last_processed_index,
                                'output_csv_filename': output_csv_filename,
                                'timestamp': str(datetime.now())
                            }, f)
                        continue

                    for role in ROLES:
                        print(f"    Generating recommendation and score for role: {role}...")
                        sys_msg_q2 = SYSTEM_TEMPLATE.format(role=role)
                        
                        model_q2 = "deepseek-chat"
                        messages_q2 = [
                            {"role": "system", "content": sys_msg_q2},
                            {"role": "user", "content": usr_msg_q1},
                            {"role": "assistant", "content": common_reasoning},
                            {"role": "user", "content": Q2_TEMPLATE},
                        ]
                        temperature_q2 = 0.0
                        max_tokens_q2 = 500
                        top_p_q2 = 1.0

                        try:
                            resp2 = client.chat.completions.create(
                                model=model_q2,
                                messages=messages_q2,
                                temperature=temperature_q2,
                                max_tokens=max_tokens_q2,
                                top_p=top_p_q2,
                            )
                            recommend_text = resp2.choices[0].message.content.strip()
                            role_score = extract_score(recommend_text, strict=False)
                            role_recommendation = extract_recommendation(recommend_text)

                            row[f"{role}_recommend"] = recommend_text.capitalize().replace("\n", "  ")
                            row[f"{role}_recommend_model"] = model_q2
                            row[f"{role}_recommend_messages"] = json.dumps(messages_q2)
                            row[f"{role}_recommend_temperature"] = temperature_q2
                            row[f"{role}_recommend_max_tokens"] = max_tokens_q2
                            row[f"{role}_score"] = role_score
                            row[f"{role}_recommendation"] = role_recommendation
                        
                        except AuthenticationError as e:
                            print(f"⚠️ Authentication failed for {role}: {e}")
                            row[f"{role}_recommend"] = f"AuthenticationError: {e}"
                            row[f"{role}_recommend_model"] = model_q2
                            row[f"{role}_recommend_messages"] = json.dumps(messages_q2)
                            row[f"{role}_recommend_temperature"] = temperature_q2
                            row[f"{role}_recommend_max_tokens"] = max_tokens_q2
                            row[f"{role}_score"] = "N/A"
                            row[f"{role}_recommendation"] = "N/A"
                        except (OpenAIError, ConnectionError, TimeoutError, Exception) as e:
                            print(f"⚠️ Error for {role}: {e}")
                            row[f"{role}_recommend"] = f"Error: {type(e).__name__}: {e}"
                            row[f"{role}_recommend_model"] = model_q2
                            row[f"{role}_recommend_messages"] = json.dumps(messages_q2)
                            row[f"{role}_recommend_temperature"] = temperature_q2
                            row[f"{role}_recommend_max_tokens"] = max_tokens_q2
                            row[f"{role}_score"] = "N/A"
                            row[f"{role}_recommendation"] = "N/A"
                        
                        time.sleep(0.5)

                    writer.writerow(row)
                    processed_rows += 1
                    
                    # Update checkpoint after each row
                    last_processed_index = i
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_processed_index': last_processed_index,
                            'output_csv_filename': output_csv_filename,
                            'timestamp': str(datetime.now())
                        }, f)
                    
                    # Flush the file to ensure writes happen immediately
                    fout.flush()
                    
                except Exception as e:
                    print(f"⚠️ Critical error processing row {i+1}: {type(e).__name__}: {e}")
                    # Log detailed information about where processing stopped
                    with open(f"error_log_{timestamp}.txt", "a") as error_log:
                        error_log.write(f"Error at row {i+1}/{total_rows}\n")
                        error_log.write(f"Row data: {row}\n")
                        error_log.write(f"Error type: {type(e).__name__}\n")
                        error_log.write(f"Error message: {e}\n")
                        error_log.write(f"Timestamp: {datetime.now()}\n\n")
                    
                    # Update checkpoint to resume from this row next time
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_processed_index': i-1,  # Go back one row to retry this one
                            'output_csv_filename': output_csv_filename,
                            'timestamp': str(datetime.now()),
                            'error': f"{type(e).__name__}: {e}"
                        }, f)
                    
                    print(f"⚠️ Processing stopped at row {i+1}/{total_rows}. Run the script again to resume.")
                    print(f"⚠️ Checkpoint saved to {checkpoint_file}")
                    print(f"⚠️ Error details saved to error_log_{timestamp}.txt")
                    break  # Exit the loop but don't terminate the program
        
        print(f"✅ Processed {processed_rows} rows this session.")
        print(f"✅ Progress: {last_processed_index + 1}/{total_rows} rows total ({((last_processed_index + 1) / total_rows) * 100:.2f}%)")
        print(f"✅ Results saved to: {output_csv_filename}")
        
        # Delete checkpoint file if all rows processed successfully
        if last_processed_index + 1 == total_rows:
            try:
                os.remove(checkpoint_file)
                print(f"🎉 All rows processed successfully! Checkpoint file removed.")
            except:
                pass
                
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        print(f"⚠️ Progress saved: {last_processed_index + 1}/{total_rows} rows processed.")
        print(f"⚠️ Run the script again to continue from row {last_processed_index + 2}.")
    except Exception as e:
        print(f"\n❌ Critical error: {type(e).__name__}: {e}")
        print(f"⚠️ Progress saved: {last_processed_index + 1}/{total_rows} rows processed.")
        print(f"⚠️ Run the script again to continue from row {last_processed_index + 2}.")

def main_only_recommend_ds(input_file):
    """
    Uses existing Q1 reasoning from a previous output file to generate only Q2 recommendations
    using the deepseek-chat model.
    
    Args:
        input_file (str): Path to the input file containing previous Q1 results
    """
    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_filename = f"recommendations_only_q2_{timestamp}.csv"
    checkpoint_file = f"checkpoint_only_q2_{timestamp}.json"

    # Track progress
    processed_rows = 0
    total_rows = 0
    last_processed_index = -1

    try:
        if not os.path.exists(input_file):
            print(f"❌ Error: File '{input_file}' not found")
            return

        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                last_processed_index = checkpoint.get('last_processed_index', -1)
                output_csv_filename = checkpoint.get('output_csv_filename', output_csv_filename)
                print(f"🔄 Resuming from row {last_processed_index + 2} using file '{output_csv_filename}'")

        # Read the input file
        df = pd.read_csv(input_file, encoding="latin1")
        all_rows = df.to_dict('records')
        total_rows = len(all_rows)
        print(f"🔍 Loaded {total_rows} rows from input file.")



        # Remove original Q2 fields from output
        q2_suffixes = [
            "_recommend", "_recommend_model", "_recommend_messages",
            "_recommend_temperature", "_recommend_max_tokens",
            "_score", "_recommendation"
        ]
        # Only keep columns that are NOT original Q2 fields
        original_columns = [
            col for col in df.columns
            if not any(col.startswith(role) and any(col.endswith(suf) for suf in q2_suffixes) for role in ROLES)
        ]
        # Add new Q2 fields
        extra_fields = []
        for role in ROLES:
            extra_fields.extend([
                f"{role}_score_new",
                f"{role}_recommendation_new",
                f"{role}_recommend_new",
                f"{role}_prompt_new",
                f"{role}_temperature_new",
                f"{role}_max_tokens_new",
                f"{role}_top_p_new"
            ])
        output_fields = original_columns + extra_fields

        # Setup output file
        file_exists = os.path.isfile(output_csv_filename) and os.path.getsize(output_csv_filename) > 0
        
        with open(output_csv_filename, "a" if file_exists else "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=output_fields)
            
            if not file_exists:
                writer.writeheader()

            # Process each row
            for i, row in enumerate(all_rows):
                if i <= last_processed_index:
                    continue

                try:
                    print(f"Processing row {i+1}/{total_rows}...")

                    # Get existing Q1 reasoning
                    common_reasoning = row.get("q1_reasoning", "")
                    if not common_reasoning or "Error" in common_reasoning or "Authentication" in common_reasoning:
                        print(f"⚠️ Invalid Q1 reasoning for row {i+1}, skipping...")
                        continue

                    # Create user message from profile data
                    usr_msg_q1 = Q1_TEMPLATE.format(
                        age=row.get("age", ""),
                        gender=row.get("gender", ""),
                        net_cash=row.get("net_cash", 0),  # <-- remove float() here
                        pct=row.get("percentage", ""),
                    )

                    # Process each role
                    for role in ROLES:
                        print(f"    Generating new recommendation for role: {role}...")
                        sys_msg_q2 = SYSTEM_TEMPLATE.format(role=role)
                        
                        model_q2 = "deepseek-chat"
                        messages_q2 = [
                            {"role": "system", "content": sys_msg_q2},
                            {"role": "user", "content": usr_msg_q1},
                            {"role": "assistant", "content": common_reasoning},
                            {"role": "user", "content": Q2_TEMPLATE},
                        ]
                        temperature_q2 = 0.0
                        max_tokens_q2 = 500
                        top_p_q2 = 1.0

                        try:
                            resp2 = client.chat.completions.create(
                                model=model_q2,
                                messages=messages_q2,
                                temperature=temperature_q2,
                                max_tokens=max_tokens_q2,
                                top_p=top_p_q2,
                            )
                            recommend_text = resp2.choices[0].message.content.strip()
                            role_score = extract_score(recommend_text, strict=False)
                            role_recommendation = extract_recommendation(recommend_text)

                            row[f"{role}_recommend_new"] = recommend_text.capitalize().replace("\n", "  ")
                            row[f"{role}_score_new"] = role_score
                            row[f"{role}_recommendation_new"] = role_recommendation
                            row[f"{role}_prompt_new"] = json.dumps(messages_q2)
                            row[f"{role}_temperature_new"] = temperature_q2
                            row[f"{role}_max_tokens_new"] = max_tokens_q2
                            row[f"{role}_top_p_new"] = top_p_q2

                        except (AuthenticationError, OpenAIError, ConnectionError, TimeoutError, Exception) as e:
                            error_msg = f"{type(e).__name__}: {e}"
                            print(f"⚠️ Error for {role}: {error_msg}")
                            row[f"{role}_recommend_new"] = f"Error: {error_msg}"
                            row[f"{role}_score_new"] = "N/A"
                            row[f"{role}_recommendation_new"] = "N/A"
                            row[f"{role}_prompt_new"] = json.dumps(messages_q2)
                            row[f"{role}_temperature_new"] = temperature_q2
                            row[f"{role}_max_tokens_new"] = max_tokens_q2
                            row[f"{role}_top_p_new"] = top_p_q2
                        
                        time.sleep(0.5)
                    # ...inside the for i, row in enumerate(all_rows): loop, just before writer.writerow(row)...
                    row = {k: v for k, v in row.items() if k in output_fields}
                    writer.writerow(row)
                    processed_rows += 1
                    
                    # Update checkpoint
                    last_processed_index = i
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_processed_index': last_processed_index,
                            'output_csv_filename': output_csv_filename,
                            'timestamp': str(datetime.now())
                        }, f)
                    
                    fout.flush()

                except Exception as e:
                    print(f"⚠️ Critical error processing row {i+1}: {type(e).__name__}: {e}")
                    with open(f"error_log_{timestamp}.txt", "a") as error_log:
                        error_log.write(f"Error at row {i+1}/{total_rows}\n")
                        error_log.write(f"Row data: {row}\n")
                        error_log.write(f"Error type: {type(e).__name__}\n")
                        error_log.write(f"Error message: {e}\n")
                        error_log.write(f"Timestamp: {datetime.now()}\n\n")
                    
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'last_processed_index': i-1,
                            'output_csv_filename': output_csv_filename,
                            'timestamp': str(datetime.now()),
                            'error': f"{type(e).__name__}: {e}"
                        }, f)
                    break

        print(f"\n✅ Processed {processed_rows} rows this session.")
        print(f"✅ Progress: {last_processed_index + 1}/{total_rows} rows ({((last_processed_index + 1) / total_rows) * 100:.2f}%)")
        print(f"✅ Results saved to: {output_csv_filename}")
        
        if last_processed_index + 1 == total_rows:
            try:
                os.remove(checkpoint_file)
                print(f"🎉 All rows processed successfully! Checkpoint file removed.")
            except:
                pass
 # --- Remove '_new' from all column names in the output file ---
        try:
            df_out = pd.read_csv(output_csv_filename, encoding="utf-8")
            df_out.rename(
                columns={col: col.replace('_new', '') for col in df_out.columns if col.endswith('_new')},
                inplace=True
            )
            df_out.to_csv(output_csv_filename, index=False, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Could not remove '_new' suffix from columns: {e}")

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        print(f"⚠️ Progress saved: {last_processed_index + 1}/{total_rows} rows processed.")
        print(f"⚠️ Run the script again to continue from row {last_processed_index + 2}.")
    except Exception as e:
        print(f"\n❌ Critical error: {type(e).__name__}: {e}")
        print(f"⚠️ Progress saved: {last_processed_index + 1}/{total_rows} rows processed.")
        print(f"⚠️ Run the script again to continue from row {last_processed_index + 2}.")

# filter_data()
# main()
# main_2() 
# main_3_ds()
# main_5()
# main_5_ds()
# main_only_recommend_ds("recommendations_main5_test.csv")  # Replace with your input file
# main_only_recommend_ds("r_recommendations_main5_20250525_120043.csv")  # Replace with your input file
recommend()