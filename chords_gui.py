# Jupyter widgets
# https://dataviz.shef.ac.uk/blog/16/06/2020/Jupyter-Widgets

# Good tutorial
# https://towardsdatascience.com/bring-your-jupyter-notebook-to-life-with-interactive-widgets-bc12e03f0916

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets
from datetime import datetime

from ChordsAPI import ChordsAPI



class chords_gui:
    def __init__(self):
        self.start_datetime_default = datetime.fromisoformat('2020-02-01')
        self.end_datetime_default = datetime.fromisoformat('2020-02-01')

        self.instrument_id = widgets.Select(
            description='Instrument ID: ',
            options=['1', '2', '3'],
            value='1',
            disabled=False,
        )
        
        
        self.start_date = widgets.DatePicker(
            description='Start Date',
            value=self.start_datetime_default,
            disabled=False
        )

        self.end_date = widgets.DatePicker(
            description='End Date',
            value=self.end_datetime_default,
            disabled=False
        )

        self.end_date = widgets.DatePicker(
            description='End Date',
            value=self.end_datetime_default,
            disabled=False
        )

        
        self.button = widgets.Button(
            description='Download File',
            disabled=False,
#             button_style='', # 'success', 'info', 'warning', 'danger' or ''
#             tooltip='Click me',
#             icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        
        self.button.on_click(self.download_csv_file)
        
        
        self.file_download_outputs = widgets.Textarea(
            value='',
#             placeholder='Type something',
            description='Output:',
            layout={'width': '90%', 'height': '100px'},
            disabled=False
        )        
        
        self.available_data_files = widgets.Select(
            options=self.get_availiable_files(),
            description='',
            disabled=False,
            layout={'width': 'initial'}
        )
        
        
        self.out = widgets.Output()

        
    def start_end_widgets(self):
        
        row_1 = widgets.HBox([self.instrument_id])
        row_2 = widgets.HBox([self.start_date, self.end_date])
        row_3 = widgets.HBox([self.button])
        row_4 = widgets.HBox([self.file_download_outputs])
        
        
        
#         gui_output = widgets.Output()

        display(row_1, row_2, row_3, row_4, self.out)


    def download_csv_file(self, passed_var):
        instrument_id = self.instrument_id.value
        start_str = self.start_date.value.strftime('%Y-%m-%d')
        end_str = self.end_date.value.strftime('%Y-%m-%d')

#         print(start_str)
#         print(end_str)
        
        domain = 'tzvolcano.chordsrt.com'
        chords_api =  ChordsAPI(domain)
        
        message = f'Downloading data for instrument id {instrument_id} for dates from {start_str} to {end_str}...'


        self.file_download_outputs.value = self.file_download_outputs.value + message + "\n"

        chords_api.get_csv_data(instrument_id, start_str, end_str)

        self.file_download_outputs.value = self.file_download_outputs.value + "Download complete\n"
        

        
        
    def get_availiable_files(self):
        local_data_dir = 'csv_files'
        from os import listdir
        from os.path import isfile, join
        files = [f for f in listdir(local_data_dir) if isfile(join(local_data_dir, f))]

        return(files)
    

    def select_data_file(self):        
        print("Available Data Files")
        
        self.available_data_files = widgets.Select(
            options=self.get_availiable_files(),
            description='',
            disabled=False,
            layout={'width': 'initial'}
        )
        
        data_files = widgets.HBox([self.available_data_files])

        display(data_files, self.out)
