import fastf1
from pathlib import Path
import logging
import pandas as pd
import os

# --- Setup ---

# Define the output directory using pathlib.Path
OUT_DIR = Path("../data/raw/race")

# Create the directory and any parent directories if they don't exist
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define a cache directory
CACHE_DIR = Path("fastf1_cache_dir")
CACHE_DIR.mkdir(exist_ok=True)

# Enable the FastF1 cache.
# This is CRITICAL for loops. It saves data locally.
try:
    fastf1.Cache.enable_cache(CACHE_DIR)
except Exception as e:
    logging.warning(f"Could not enable FastF1 cache: {e}")
    logging.warning("Downloads may be slow and repeated.")

# Define the year you want to fetch
YEAR = 2024

# --- Get the schedule ---
# This gets the schedule for the specified year
schedule = fastf1.get_event_schedule(YEAR)

print(f"--- Starting to fetch {YEAR} Race Results ---")
print(f"Output directory: {OUT_DIR.resolve()}") # Shows the full, absolute path

# --- Loop through each event in the schedule ---
# We use schedule.iterrows() to get each event (row)
for index, event in schedule.iterrows():
    
    event_name = event['EventName']
    round_number = event['RoundNumber']

    print(f"\nProcessing: Round {round_number} - {event_name} (Race)")

    try:
        # Get the race session (using 'R')
        race_session = fastf1.get_session(YEAR, event_name, 'R')
        
        # Load the session data.
        race_session.load(telemetry=False, weather=False, messages=False)
        
        # Check if results exist
        if race_session.results is None:
            print(f"    -> No results found for {event_name} Race.")
            continue # Skip to the next event in the loop

        # --- Define the columns you want to save ---
        # These are different columns than qualifying
        columns_to_save = [
            'Position', 'FullName', 'TeamName', 
            'Status', 'Points', 'Laps', 'GridPosition'
        ]
        
        # Get just the columns we want and make a copy
        event_results = race_session.results[columns_to_save].copy()
        
        # --- Add the Event data (the "Context") ---
        event_results['Year'] = YEAR
        event_results['RoundNumber'] = round_number
        event_results['EventName'] = event_name
        
        # Re-order columns to be more logical
        final_columns = [
            'Year', 'RoundNumber', 'EventName', 
            'Position', 'FullName', 'TeamName', 
            'GridPosition', 'Status', 'Points', 'Laps'
        ]
        event_results = event_results[final_columns]
        
        # --- Define the output file (inside the loop) ---
        # Create a clean, sortable filename
        filename = f"{YEAR}_Round_{str(round_number).zfill(2)}_{event_name.replace(' ', '_')}_Race.csv"
        out_path = OUT_DIR / filename
        
        # --- Save the results for THIS event to a CSV file ---
        event_results.to_csv(out_path, index=False)
        
        print(f"    -> Successfully saved: {out_path.name}")

    except Exception as e:
        # Catch any errors, e.g., session data not available yet
        print(f"    -> ERROR processing {event_name}: {e}")

print("\n--- All Race events processed. ---")
print("--- Script Finished. ---")