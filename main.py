import os
import sys

# Application path is the project root (this file's directory)
APP_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(APP_PATH)
sys.path.append(os.path.join(APP_PATH, 'scripts'))

# Import updated modules
from make_stations import main as step1_prepare
from extract_nc import run_extraction as step2_extract

def run_pipeline():
    print(f"Starting Gapfill Pipeline at: {APP_PATH}")
    
    # Change current working directory to APP_PATH 
    # to ensure all relative paths in scripts behave correctly
    os.chdir(APP_PATH)

    print("\n--- STEP 1: PREPARING STATION COORDINATES ---")
    step1_prepare()

    print("\n--- STEP 2: EXTRACTING NC DATA (IO INTENSIVE) ---")
    step2_extract()

    print("\n--- FINISHED EXTRACTION ---")

if __name__ == "__main__":
    run_pipeline()