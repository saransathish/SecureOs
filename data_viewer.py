import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkbootstrap import Frame, Notebook, Button, Entry, Scrollbar
from tkinterdnd2 import DND_FILES

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DataViewerApp:
    def __init__(self, master, dataframe_library):
        self.master = master
        self.dataframe_library = dataframe_library
        self.master.title("Data Viewer Application")
        
        # Create main container
        self.main_frame = Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.notebook = Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create Data Browser Tab
        self.create_data_browser_tab()
        
        # Create Data Analysis Tab
        self.create_data_analysis_tab()
        
        # Setup drag and drop
        self.setup_drag_and_drop()
        
        # Current selected dataframe
        self.current_df = None
        self.current_df_name = None
        
        # Log successful initialization
        logging.info("DataViewerApp initialized successfully with {} datasets".format(len(dataframe_library)))

    def setup_drag_and_drop(self):
        """Configure drag and drop functionality"""
        self.master.drop_target_register(DND_FILES)
        self.master.dnd_bind('<<Drop>>', self.drop_file)

    def drop_file(self, event):
        """Handle file drop event"""
        file_path = event.data
        # Clean the file path (remove {} and extra spaces)
        file_path = file_path.strip('{}')
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format. Please use .csv or .xlsx files.")
                return
                
            # Add to dataframe library
            file_name = os.path.basename(file_path)
            self.dataframe_library[file_name] = df
            self.update_data_list()
            messagebox.showinfo("Success", f"File {file_name} loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
            logging.error(f"Error loading dropped file: {str(e)}", exc_info=True)

    def create_data_browser_tab(self):
        """Create the Data Browser tab"""
        self.browser_tab = Frame(self.notebook)
        self.notebook.add(self.browser_tab, text="Data Browser")
        
        # Split into two frames
        self.left_frame = Frame(self.browser_tab, width=200)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.right_frame = Frame(self.browser_tab)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side: Dataset list
        self.list_label = ttk.Label(self.left_frame, text="Available Datasets:")
        self.list_label.pack(anchor="w", pady=(0, 5))
        
        # Search box
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_datasets)
        self.search_entry = Entry(self.left_frame, textvariable=self.search_var)
        self.search_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Dataset listbox
        self.data_listbox_frame = Frame(self.left_frame)
        self.data_listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.data_listbox = tk.Listbox(self.data_listbox_frame)
        self.data_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.listbox_scrollbar = Scrollbar(self.data_listbox_frame, command=self.data_listbox.yview)
        self.listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_listbox.config(yscrollcommand=self.listbox_scrollbar.set)
        
        self.data_listbox.bind('<<ListboxSelect>>', self.on_dataset_select)
        
        # Buttons
        self.load_btn = Button(self.left_frame, text="Load External File", command=self.load_external_file)
        self.load_btn.pack(fill=tk.X, pady=5)
        
        # Right side: Dataset view
        self.data_view_frame = Frame(self.right_frame)
        self.data_view_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info frame at the top of data view
        self.info_frame = Frame(self.data_view_frame)
        self.info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.dataset_name_label = ttk.Label(self.info_frame, text="No dataset selected")
        self.dataset_name_label.pack(anchor="w")
        
        self.dataset_info_label = ttk.Label(self.info_frame, text="")
        self.dataset_info_label.pack(anchor="w")
        
        # Table view
        self.table_frame = Frame(self.data_view_frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial list update
        self.update_data_list()

    def create_data_analysis_tab(self):
        """Create the Data Analysis tab"""
        self.analysis_tab = Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Data Analysis")
        
        # Split into control panel and visualization area
        self.control_frame = Frame(self.analysis_tab, width=200)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.visualization_frame = Frame(self.analysis_tab)
        self.visualization_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel widgets - will be populated when a dataset is selected
        self.analysis_dataset_label = ttk.Label(self.control_frame, text="No dataset selected for analysis")
        self.analysis_dataset_label.pack(anchor="w", pady=(0, 10))
        
        # Placeholder for visualization
        self.fig_placeholder = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas_placeholder = FigureCanvasTkAgg(self.fig_placeholder, master=self.visualization_frame)
        self.canvas_placeholder.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_data_list(self):
        """Update the list of available datasets"""
        self.data_listbox.delete(0, tk.END)
        search_term = self.search_var.get().lower()
        
        for i, name in enumerate(sorted(self.dataframe_library.keys())):
            if search_term in name.lower():
                self.data_listbox.insert(tk.END, name)
        
        logging.info(f"Data list updated with {self.data_listbox.size()} items")

    def filter_datasets(self, *args):
        """Filter the dataset list based on search term"""
        self.update_data_list()

    def on_dataset_select(self, event):
        """Handle dataset selection from the list"""
        selection = self.data_listbox.curselection()
        if not selection:
            return
        
        selected_item = self.data_listbox.get(selection[0])
        self.display_dataset(selected_item)

    def display_dataset(self, dataset_name):
        """Display the selected dataset in the table view"""
        try:
            # Clear existing widgets in table frame
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            
            self.current_df_name = dataset_name
            self.current_df = self.dataframe_library[dataset_name]
            
            # Update info labels
            self.dataset_name_label.config(text=f"Dataset: {dataset_name}")
            shape_info = f"Shape: {self.current_df.shape[0]} rows × {self.current_df.shape[1]} columns"
            dtype_info = "Data types: " + ", ".join([f"{col}: {dtype}" for col, dtype in 
                                                    zip(self.current_df.dtypes.index[:3], 
                                                       self.current_df.dtypes.values[:3])])
            if len(self.current_df.dtypes) > 3:
                dtype_info += "..."
                
            self.dataset_info_label.config(text=f"{shape_info} | {dtype_info}")
            
            # Create treeview for data display
            self.create_treeview()
            
            # Update analysis tab
            self.update_analysis_tab()
            
            logging.info(f"Displayed dataset: {dataset_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying dataset: {str(e)}")
            logging.error(f"Error displaying dataset {dataset_name}: {str(e)}", exc_info=True)

    def create_treeview(self):
        """Create a treeview to display the dataframe"""
        # Create a frame with scrollbars
        frame = Frame(self.table_frame)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Set up scrollbars
        v_scrollbar = Scrollbar(frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = Scrollbar(frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create treeview
        columns = list(self.current_df.columns)
        tree = ttk.Treeview(frame, columns=columns, show='headings',
                          yscrollcommand=v_scrollbar.set,
                          xscrollcommand=h_scrollbar.set)
        
        # Set scrollbar commands
        v_scrollbar.config(command=tree.yview)
        h_scrollbar.config(command=tree.xview)
        
        # Set column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        
        # Insert data rows (limit to first 1000 rows for performance)
        display_df = self.current_df.head(1000)
        for i, row in display_df.iterrows():
            values = [str(val) if not pd.isna(val) else "" for val in row]
            tree.insert('', 'end', values=values)
            
        # If more than 1000 rows, add a note
        if len(self.current_df) > 1000:
            tree.insert('', 'end', values=["... Only first 1000 rows shown"] + 
                       ["" for _ in range(len(columns)-1)])
        
        tree.pack(fill=tk.BOTH, expand=True)

    def update_analysis_tab(self):
        """Update the analysis tab with controls for the selected dataset"""
        # Clear existing widgets
        for widget in self.control_frame.winfo_children():
            widget.destroy()
            
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
        
        if self.current_df is None:
            self.analysis_dataset_label = ttk.Label(self.control_frame, text="No dataset selected for analysis")
            self.analysis_dataset_label.pack(anchor="w", pady=(0, 10))
            return
        
        # Dataset info
        ttk.Label(self.control_frame, text=f"Dataset: {self.current_df_name}").pack(anchor="w", pady=(0, 5))
        
        # Analysis type selection
        ttk.Label(self.control_frame, text="Select Analysis Type:").pack(anchor="w", pady=(10, 5))
        
        analysis_types = ["Summary Statistics", "Histogram", "Scatter Plot", "Box Plot"]
        self.analysis_var = tk.StringVar(value=analysis_types[0])
        
        for analysis in analysis_types:
            ttk.Radiobutton(self.control_frame, text=analysis, value=analysis, 
                          variable=self.analysis_var, command=self.update_analysis_options).pack(anchor="w")
        
        # Placeholder for column selection (will be populated by update_analysis_options)
        self.column_frame = Frame(self.control_frame)
        self.column_frame.pack(fill=tk.X, pady=10)
        
        # Run button
        Button(self.control_frame, text="Run Analysis", command=self.run_analysis).pack(pady=10)
        
        # Initialize options
        self.update_analysis_options()

    def update_analysis_options(self):
        """Update the analysis options based on selected analysis type"""
        # Clear existing widgets
        for widget in self.column_frame.winfo_children():
            widget.destroy()
        
        analysis_type = self.analysis_var.get()
        
        numeric_columns = self.current_df.select_dtypes(include=np.number).columns.tolist()
        all_columns = self.current_df.columns.tolist()
        
        if analysis_type == "Summary Statistics":
            ttk.Label(self.column_frame, text="No column selection needed").pack(anchor="w")
            
        elif analysis_type == "Histogram":
            ttk.Label(self.column_frame, text="Select Column:").pack(anchor="w")
            self.hist_var = tk.StringVar(value=numeric_columns[0] if numeric_columns else "")
            ttk.Combobox(self.column_frame, textvariable=self.hist_var, 
                       values=numeric_columns).pack(fill=tk.X)
            
        elif analysis_type == "Scatter Plot":
            ttk.Label(self.column_frame, text="X-Axis:").pack(anchor="w")
            self.x_var = tk.StringVar(value=numeric_columns[0] if numeric_columns else "")
            ttk.Combobox(self.column_frame, textvariable=self.x_var, 
                       values=numeric_columns).pack(fill=tk.X)
            
            ttk.Label(self.column_frame, text="Y-Axis:").pack(anchor="w")
            self.y_var = tk.StringVar(value=numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0] if numeric_columns else "")
            ttk.Combobox(self.column_frame, textvariable=self.y_var, 
                       values=numeric_columns).pack(fill=tk.X)
            
        elif analysis_type == "Box Plot":
            ttk.Label(self.column_frame, text="Select Column:").pack(anchor="w")
            self.box_var = tk.StringVar(value=numeric_columns[0] if numeric_columns else "")
            ttk.Combobox(self.column_frame, textvariable=self.box_var, 
                       values=numeric_columns).pack(fill=tk.X)
            
            ttk.Label(self.column_frame, text="Group By (optional):").pack(anchor="w")
            self.group_var = tk.StringVar(value="")
            ttk.Combobox(self.column_frame, textvariable=self.group_var, 
                       values=[""] + all_columns).pack(fill=tk.X)

    def run_analysis(self):
        """Run the selected analysis and display results"""
        try:
            analysis_type = self.analysis_var.get()
            
            # Clear visualization frame
            for widget in self.visualization_frame.winfo_children():
                widget.destroy()
            
            # Create figure and canvas
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            if analysis_type == "Summary Statistics":
                # Create text widget to display summary
                text_widget = tk.Text(self.visualization_frame, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True)
                
                # Get summary statistics
                summary = self.current_df.describe().T
                text_widget.insert(tk.END, summary.to_string())
                
            elif analysis_type == "Histogram":
                column = self.hist_var.get()
                if not column:
                    messagebox.showerror("Error", "Please select a column")
                    return
                    
                # Create histogram
                self.current_df[column].hist(ax=ax, bins=20)
                ax.set_title(f'Histogram of {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                
                canvas = FigureCanvasTkAgg(fig, self.visualization_frame)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas.draw()
                
            elif analysis_type == "Scatter Plot":
                x_col = self.x_var.get()
                y_col = self.y_var.get()
                
                if not x_col or not y_col:
                    messagebox.showerror("Error", "Please select both X and Y columns")
                    return
                
                # Create scatter plot
                ax.scatter(self.current_df[x_col], self.current_df[y_col])
                ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                
                canvas = FigureCanvasTkAgg(fig, self.visualization_frame)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas.draw()
                
            elif analysis_type == "Box Plot":
                column = self.box_var.get()
                group = self.group_var.get()
                
                if not column:
                    messagebox.showerror("Error", "Please select a column")
                    return
                
                # Create box plot
                if group:
                    # Group by another column
                    grouped_data = [self.current_df[self.current_df[group] == val][column].dropna() 
                                  for val in self.current_df[group].unique()]
                    ax.boxplot(grouped_data)
                    ax.set_xticklabels(self.current_df[group].unique())
                    ax.set_title(f'Box Plot of {column} grouped by {group}')
                else:
                    # Simple box plot
                    ax.boxplot(self.current_df[column].dropna())
                    ax.set_title(f'Box Plot of {column}')
                
                ax.set_ylabel(column)
                
                canvas = FigureCanvasTkAgg(fig, self.visualization_frame)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas.draw()
            
            logging.info(f"Analysis {analysis_type} completed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error running analysis: {str(e)}")
            logging.error(f"Error running analysis: {str(e)}", exc_info=True)

    def load_external_file(self):
        """Load an external file into the dataframe library"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format. Please use .csv or .xlsx files.")
                return
            
            # Add to dataframe library
            file_name = os.path.basename(file_path)
            self.dataframe_library[file_name] = df
            self.update_data_list()
            messagebox.showinfo("Success", f"File {file_name} loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
            logging.error(f"Error loading external file: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Simple test code
    root = tk.Tk()
    test_df = pd.DataFrame({
        'A': range(100),
        'B': [i * 2 for i in range(100)],
        'C': [f'Item {i}' for i in range(100)]
    })
    
    test_library = {
        'test_data': test_df
    }
    
    app = DataViewerApp(root, test_library)
    root.mainloop()