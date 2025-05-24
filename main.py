import os
import pandas as pd
import tkinter as tk
from tkinter import ttk
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.csv as pv
import hashlib
import pickle
import logging
from data_viewer import DataViewerApp
import ttkbootstrap as ttkb
from tkinterdnd2 import TkinterDnD

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


data_source_library = {
    "uk_monthly_police_reporting": "Documents/DATA 2/CY Crime/sector_level_crime.csv",  # done
    "schools": "Documents/DATA 2/Schools/schools_for_script.csv",  # done
    "uk_retail_parks_&_shopping_centres": "Documents/DATA 2/Retail & Shopping Parks/retail_shopping_park_locations.csv",  # done
    "uk_major_junction": "Documents/DATA 2/Junctions/Junctions.xlsx",  # done
    "uk_transport_hubs": "Documents/DATA 2/Transport Hubs/England_Scotland_Wales_transport_hubs.csv",  # done
    "uk_wellbeing": "Documents/DATA 2/Wellbeing/uk_wellbeing_for_script.csv",  # done
    "uk_unemployment": "Documents/DATA 2/Unemployment_Student/uk_unemployment_for_script.csv",  # done
    "uk_historic_detailed_police_reporting": "Documents/DATA 2/Historic Crime/uk_historic_crime_data_for_script.xlsx",  # done
    "scottish_monthly_police_reporting": "Documents/DATA 2/CY Crime/Police Scotland/Scotland_CY_Crime.csv",  # NOT DONE
    "uk_population_density": "Documents/DATA 2/Population Density/uk_population_density_for_script.csv",  # done
    "scotland_population_density": "Documents/DATA 2/Population Density/scotland_population_density_for_script.csv",  # done
    "uk_student_population": "Documents/DATA 2/Unemployment_Student/uk_student_population_for_script.csv",  # done
    "scotland_student_population": "Documents/DATA 2/Unemployment_Student/scotland_student_population_for_script.csv",  # done
    "scotland_unemployment": "Documents/DATA 2/Unemployment_Student/scotland_unemployment_for_script.csv",  # done
    "scotland_historic_detailed_police_reporting": "Documents/DATA 2/Historic Crime/scotland_historic_crime_data_for_script.csv",
    "ni_monthly_police_reporting": "Documents/DATA 2/CY Crime/ni_monthly_crime_for_script.csv",
    "ni_population_density": "Documents/DATA 2/Population Density/ni_population_density_for_script.csv",  # done
    "ni_wellbeing": "Documents/DATA 2/Wellbeing/ni_wellbeing_for_script.csv",  # done
    "ni_student_population": "Documents/DATA 2/Unemployment_Student/ni_student_population_for_script.csv",  # done
    "ni_unemployment": "Documents/DATA 2/Unemployment_Student/ni_unemployment_for_script.csv",  # done
    "pubs": "Documents/DATA 2/Pubs/pubs_for_script.csv",
    "scottish_postcode_directory": "Documents/DATA 2/Historic Crime/scottish_postcode_directory.csv"
}



class LoadingScreen:
    def __init__(self, master):
        self.master = master
        self.frame = ttk.Frame(self.master)
        self.frame.pack(expand=True, fill='both')

        self.progress = ttk.Progressbar(
            self.frame, length=250, mode='determinate')
        self.progress.pack(pady=20)

        self.status_label = ttk.Label(self.frame, text="Loading...")
        self.status_label.pack()

    def update_progress(self, value, status):
        self.progress['value'] = value
        self.status_label.config(text=status)
        self.master.update_idletasks()

    def destroy(self):
        self.frame.destroy()


def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_file(name, path, cache_dir):
    try:
        # full_path = rf"C:\Users\{os.getlogin()}\ASEL\{path}"
        full_path = rf"/Users/saran/{path}"
        print('hello - ',full_path , 'end')

        logging.info(f"Attempting to load {name} from {full_path}")

        if not os.path.exists(full_path):
            logging.error(f"File not found: {full_path}")
            return name, None

        file_extension = os.path.splitext(full_path)[1].lower()
        logging.info(f"File extension for {name}: {file_extension}")

        if file_extension == '.csv':
            logging.info(f"Reading CSV file: {name}")
            table = pv.read_csv(full_path)
            df = table.to_pandas()
        elif file_extension == '.xlsx':
            logging.info(f"Reading Excel file: {name}")
            df = pd.read_excel(full_path)
        elif file_extension == '.ods':
            logging.info(f"Reading ODS file: {name}")
            df = pd.read_excel(full_path, engine='odf')
        else:
            logging.error(f"Unsupported file format for {
                          name}: {file_extension}")
            return name, None

        if df.empty:
            logging.warning(f"Loaded dataframe for {name} is empty")
        else:
            logging.info(f"Successfully loaded {name}. Shape: {df.shape}")

        return name, df
    except Exception as e:
        logging.error(f"Error loading {name} from {
                      path}: {str(e)}", exc_info=True)
        return name, None


def load_datasets(data_source_library, loading_screen):
    dataframe_library = {}
    total_files = len(data_source_library)
    cache_dir = os.path.join(os.path.expanduser("~"), ".data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    logging.info(f"Starting to load {total_files} datasets")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_name = {executor.submit(load_file, name, path, cache_dir): name
                          for name, path in data_source_library.items()}

        for i, future in enumerate(as_completed(future_to_name)):
            name = future_to_name[future]
            try:
                name, df = future.result()
                if df is not None and not df.empty:
                    dataframe_library[name] = df
                    logging.info(f"Successfully added {
                                 name} to dataframe_library. Shape: {df.shape}")
                else:
                    logging.warning(
                        f"Dataset {name} is empty or failed to load")
            except Exception as e:
                logging.error(f"Error processing {name}: {
                              str(e)}", exc_info=True)

            loading_screen.update_progress(
                (i + 1) / total_files * 100, f"Loaded {i + 1}/{total_files} files")

    logging.info(f"Completed loading. Successfully loaded {len(dataframe_library)} out of {total_files} datasets")
    loading_screen.update_progress(100, "Loading complete!")
    return dataframe_library


def start_application():
    root = TkinterDnD.Tk()  # Use TkinterDnD.Tk instead of ttkb.Window
    style = ttkb.Style(theme="cosmo")
    root.title("Data Loader and Viewer")
    root.geometry("800x600")

    loading_screen = LoadingScreen(root)

    def load_data_thread():
        global new_library
        new_library = load_datasets(data_source_library, loading_screen)
        root.after(1000, lambda: show_data_viewer(
            root, new_library, loading_screen))

    thread = Thread(target=load_data_thread)
    thread.start()

    root.mainloop()


def show_data_viewer(root, data_library, loading_screen):
    loading_screen.destroy()
    DataViewerApp(root, data_library)


if __name__ == "__main__":
    start_application()

__all__ = ["start_application", "show_data_viewer", "load_datasets"]