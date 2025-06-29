import pandas as pd
import csv
import time
import os
import certifi
# from openai import AuthenticationError, OpenAIError, OpenAI # OLD
from openai import AuthenticationError, OpenAIError, AsyncOpenAI # NEW
from datetime import datetime # Added import
import json # Added import for dumping messages
import re # Added import for regular expressions
import asyncio # NEW
from analyze_scores import analyze_role_scores
from filter_data import filter_random_data_by_product_type
from main import only_q1_batch, only_q2_batch


FOLDER_NAME = "groups/group2/" # Folder name for output files


# ─── Use certifi's CA bundle for all HTTPS requests ─────────────────────────
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"]     = certifi.where()

# ─── Load your OpenAI key from the environment ─────────────────────────────
TEST_MODEL = "gpt-4o-mini"  # Or "deepseek-chat", etc.

if TEST_MODEL == "gpt-4o-mini":
    hardcoded_api_key = "REMOVED"
    api_base_url = "https://api1.oaipro.com/v1"
elif TEST_MODEL == "deepseek-chat":
    hardcoded_api_key="REMOVED"
    api_base_url = "https://api.deepseek.com"
else:
    raise ValueError(f"Unsupported model: {TEST_MODEL}")


client = AsyncOpenAI( 
    api_key=hardcoded_api_key,
    base_url=api_base_url
)

# ─── Input CSV file ────────────────────────────────────────────────────────
Q1_INPUT_CSV = filter_random_data_by_product_type(1, ["crypto", "stock"], FOLDER_NAME) # Generate a random dataset with 1 row for testing

    
async def run_Q1_and_close_client():
    if os.path.exists(Q1_INPUT_CSV):
        try:
            q1_out_put_file=await only_q1_batch(Q1_INPUT_CSV, model=TEST_MODEL)
        
        finally:
            await client.close() 
        return q1_out_put_file

    else:
        print(f"❌ Input file for processing not found in process Q1: {Q1_INPUT_CSV}")

Q2_INPUT_CSV = asyncio.run(run_Q1_and_close_client())

async def run_Q2_and_close_client():
    if os.path.exists(Q2_INPUT_CSV):
        try:
            q2_out_put_file=await only_q2_batch(Q2_INPUT_CSV, model=TEST_MODEL)
        
        finally:
            await client.close() 
        return q2_out_put_file

    else:
        print(f"❌ Input file for processing not found in process Q2: {Q2_INPUT_CSV}")

q2_out_put_file=asyncio.run(run_Q2_and_close_client())

# ─── Analyze the scores ─────────────────────────────────────────────────────
if os.path.exists(q2_out_put_file):
    try:
        analyze_role_scores(q2_out_put_file, roles=ROLES, output_folder_path=FOLDER_NAME)
    except Exception as e:
        print(f"❌ Error analyzing role scores: {type(e).__name__}: {e}")

print(f"✅ complete. Results saved in folder: {FOLDER_NAME}")