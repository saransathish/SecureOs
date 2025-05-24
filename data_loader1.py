import os
import pandas as pd
import tkinter as tk
from tkinter import ttk
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.csv as pv
import hashlib
import pickle
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    _instance = None
    _lock = Lock()
    _initialized = False
    
    # Define paths relative to the DATA 2 folder
    data_source_library = {
        "uk_monthly_police_reporting": r"CY Crime/sector_level_crime.csv",
        "schools": r"Schools/schools_for_script.csv",
        "uk_retail_parks_&_shopping_centres": r"Retail & Shopping Parks/retail_shopping_park_locations.csv",
        "uk_major_junction": r"Junctions/Junctions.xlsx",
        "uk_transport_hubs": r"Transport Hubs/England_Scotland_Wales_transport_hubs.csv",
        "uk_wellbeing": r"Wellbeing/uk_wellbeing_for_script.csv",
        "uk_unemployment": r"Unemployment_Student/uk_unemployment_for_script.csv",
        "uk_historic_detailed_police_reporting": r"Historic Crime/uk_historic_crime_data_for_script.xlsx",
        "scottish_monthly_police_reporting": r"CY Crime/Police Scotland/Scotland_CY_Crime.csv",
        "uk_population_density": r"Population Density/uk_population_density_for_script.csv",
        "scotland_population_density": r"Population Density/scotland_population_density_for_script.csv",
        "uk_student_population": r"Unemployment_Student/uk_student_population_for_script.csv",
        "scotland_student_population": r"Unemployment_Student/scotland_student_population_for_script.csv",
        "scotland_unemployment": r"Unemployment_Student/scotland_unemployment_for_script.csv",
        "scotland_historic_detailed_police_reporting": r"Historic Crime/scotland_historic_crime_data_for_script.csv",
        "ni_monthly_police_reporting": r"CY Crime/ni_monthly_crime_for_script.csv",
        "ni_population_density": r"Population Density/ni_population_density_for_script.csv",
        "ni_wellbeing": r"Wellbeing/ni_wellbeing_for_script.csv",
        "ni_student_population": r"Unemployment_Student/ni_student_population_for_script.csv",
        "ni_unemployment": r"Unemployment_Student/ni_unemployment_for_script.csv",
        "pubs": r"Pubs/pubs_for_script.csv",
        "scottish_postcode_directory": r"Historic Crime/scottish_postcode_directory.csv", 
        "population": r"Postcodes/ukpostcodes.csv", 
        "uk_homelessness": r"Homelessness/uk_homelessness_for_script.csv",
        "scotland_homelessness": r"Homelessness/scotland_homelessness_for_script.csv", 
        "ni_homelessness": r"Homelessness/ni_homelessness_for_script.csv"
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    # Get the directory where the script is located
                    self.script_dir = os.path.dirname(os.path.abspath(__file__))
                    # Path to DATA 2 folder relative to script location
                    self.data_dir = os.path.join(self.script_dir, "DATA 2")
                    self.dataframe_library = {}
                    self._load_datasets()
                    self._initialized = True

    def _load_file(self, name, rel_path, cache_dir):
        try:
            # Build full path by joining script directory, DATA 2 folder, and relative path
            full_path = os.path.join(self.data_dir, rel_path)
            full_path = os.path.normpath(full_path)  # Normalize path for cross-platform compatibility
            
            logging.info(f"Attempting to load {name} from {full_path}")

            if not os.path.exists(full_path):
                logging.error(f"File not found: {full_path}")
                return name, None

            file_extension = os.path.splitext(rel_path)[1].lower()
            cache_file = os.path.join(cache_dir, f"{name}.pkl")

            # Load data based on file type
            logging.info(f"Loading {name} from {full_path}")
            if file_extension == '.csv':
                df = pv.read_csv(full_path).to_pandas()
            elif file_extension == '.xlsx':
                df = pd.read_excel(full_path)
            elif file_extension == '.ods':
                df = pd.read_excel(full_path, engine='odf')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            logging.info(f"Successfully loaded {name}. Shape: {df.shape}")
            return name, df
        except Exception as e:
            logging.error(f"Error loading {name} from {rel_path}: {str(e)}", exc_info=True)
            return name, None

    def _load_datasets(self):
        cache_dir = os.path.join(os.path.expanduser("~"), ".data_cache")
        os.makedirs(cache_dir, exist_ok=True)

        logging.info("Starting to load datasets")
        logging.info(f"Looking for data in: {self.data_dir}")

        # Check if DATA 2 directory exists
        if not os.path.exists(self.data_dir):
            logging.error(f"DATA 2 directory not found at: {self.data_dir}")
            raise FileNotFoundError(f"DATA 2 directory not found at: {self.data_dir}")

        for name, path in self.data_source_library.items():
            name, df = self._load_file(name, path, cache_dir)
            if df is not None and not df.empty:
                self.dataframe_library[name] = df
                logging.info(f"Successfully loaded {name}. Shape: {df.shape}")
            else:
                logging.warning(f"Dataset {name} is empty or failed to load")

        logging.info("Completed loading datasets")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DataLoader()
        return cls._instance

    def get_data(self):
        return self.dataframe_library

# Convenience functions for accessing the data
def get_data():
    return DataLoader.get_instance().get_data()

def get_data_source_library():
    return DataLoader.data_source_library

# For backwards compatibility
data_source_library = DataLoader.data_source_library
# print(data_source_library)

if __name__ == "__main__":
    obj = DataLoader()
    obj._load_datasets()