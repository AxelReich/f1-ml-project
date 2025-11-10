import fastf1
import pandas as pd
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# create cache for future uses, and to check if the info is already there
# CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'f1_cache')
# if not os.path.exists(CACHE_DIR):
#     os.makedirs(CACHE_DIR)
# fastf1.Cache.enable_cache(CACHE_DIR)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def fetch_season_data(year: int):
    """
    Fetches all Race Results, Qualifying Results, and Pit Stop data
    for a given F1 season.

    Saves the data as CSV files in the data/raw/ directory.

    Args:
        year: The championship season year (e.g., 2023)
    """
    logging.info(f"Fetching f1 data for {year} year")

    # get schedule, name of the carrer
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        logging.error(f"Could not fetch event schedule for {year}. Error: {e}")
        return

    # We only want to process events that have already happened
    # In a real-world scenario, you'd check against the current date
    # For historical data, we'll just process all of them
    
    # Use schedule['EventName'] for iterating
    race_events = schedule[schedule['EventFormat'] != 'testing']['EventName']
    
    all_race_results = []
    all_q_results = []
    all_pit_stops = []

    for event_name in race_events:
        try:
            logging.info(f"Fetching data for: {year} {event_name}")
            
            # Use R for race and Q for qualy 
            session_r = fastf1.get_session(year, event_name, 'R')
            session_r.load() # Load all data (laps, telemetry, etc.)
            
            # results 
            race_results = session_r.results
            if race_results is not None and not race_results.empty:
                race_results['EventName'] = event_name
                race_results['Year'] = year
                all_race_results.append(race_results)
            else:
                logging.warning(f"No race results found for {event_name}")

            # pit stop 
            pit_stops = session_r.pitstops
            if pit_stops is not None and not pit_stops.empty:
                pit_stops['EventName'] = event_name
                pit_stops['Year'] = year
                all_pit_stops.append(pit_stops)
            else:
                logging.warning(f"No pit stop data found for {event_name}")

            # qualy times
            session_q = fastf1.get_session(year, event_name, 'Q')
            session_q.load()
            
            q_results = session_q.results
            if q_results is not None and not q_results.empty:
                q_results['EventName'] = event_name
                q_results['Year'] = year
                all_q_results.append(q_results)
            else:
                logging.warning(f"No qualifying results found for {event_name}")

            logging.info(f"Successfully processed {event_name}")

        except Exception as e:
            # Some sessions might not exist (e.g., canceled races)
            # or data might not be available
            logging.error(f"Could not load data for {event_name}. Error: {e}", exc_info=True)
            
    
    if all_race_results:
        df_race_results = pd.concat(all_race_results)
        race_file = os.path.join(OUTPUT_DIR, f"{year}_race_results.csv")
        df_race_results.to_csv(race_file, index=False)
        logging.info(f"Saved all race results to {race_file}")

    if all_q_results:
        df_q_results = pd.concat(all_q_results)
        q_file = os.path.join(OUTPUT_DIR, f"{year}_qualifying_results.csv")
        df_q_results.to_csv(q_file, index=False)
        logging.info(f"Saved all qualifying results to {q_file}")

    if all_pit_stops:
        df_pit_stops = pd.concat(all_pit_stops)
        pits_file = os.path.join(OUTPUT_DIR, f"{year}_pit_stops.csv")
        df_pit_stops.to_csv(pits_file, index=False)
        logging.info(f"Saved all pit stops to {pits_file}")

    logging.info(f"--- Finished data fetch for {year} season ---")


if __name__ == "__main__":
    for year in [2022]:
        fetch_season_data(year)