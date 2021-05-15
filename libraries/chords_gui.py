# Jupyter widgets
# https://dataviz.shef.ac.uk/blog/16/06/2020/Jupyter-Widgets

# Good tutorial
# https://towardsdatascience.com/bring-your-jupyter-notebook-to-life-with-interactive-widgets-bc12e03f0916

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets

from datetime import datetime
from datetime import timedelta

from ChordsAPI import ChordsAPI

import pandas as pd
import os


class chords_gui:
    def __init__(self, domain):
        
        self.domain = domain
        self.local_data_dir = 'csv_files'
        os.makedirs(self.local_data_dir, exist_ok=True)
        
        self.start_datetime_default = datetime.now() - timedelta(days=1)
        self.end_datetime_default = datetime.now() - timedelta(days=1)

        self.instrument_id = widgets.Select(
            description='Instrument ID: ',
            options=['1', '2', '4', '5', '6', '7', '8', '9'],
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
        )
        
        self.button.on_click(self.download_csv_file)
        
        
        self.file_download_outputs = widgets.Textarea(
            value='',
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

        
    def start_end_widgets(self, start_date_str = '', end_date_str = ''):
        
        if not start_date_str == '':
            self.start_date.value = datetime.fromisoformat(start_date_str)

        if not end_date_str == '':
            self.end_date.value = datetime.fromisoformat(end_date_str)


        row_1 = widgets.HBox([self.instrument_id])
        row_2 = widgets.HBox([self.start_date, self.end_date])
        row_3 = widgets.HBox([self.button])
        row_4 = widgets.HBox([self.file_download_outputs])
        
        
        display(row_1, row_2, row_3, row_4, self.out)


    def download_csv_file(self, passed_var = ''):
        instrument_id = self.instrument_id.value
        start_str = self.start_date.value.strftime('%Y-%m-%d')
        end_str = self.end_date.value.strftime('%Y-%m-%d')        
        
        chords_api =  ChordsAPI(self.domain)
        
        message = f'Downloading data for instrument id {instrument_id} for dates from {start_str} to {end_str}...'

        self.file_download_outputs.value = self.file_download_outputs.value + message + "\n"

        chords_api.get_csv_data(instrument_id, start_str, end_str)

        self.file_download_outputs.value = self.file_download_outputs.value + "Download complete\n"
        

        
        
    def get_availiable_files(self):
        from os import listdir
        from os.path import isfile, join

        files = []
        for file in os.listdir(self.local_data_dir):
            if isfile(join(self.local_data_dir, file)):
                if file.endswith(".csv"):
                    files.append(file)

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
        

        
    def load_data_from_file(self, file_name):
        file_path = f'{self.local_data_dir}/{file_name}'
        print(file_path)

        return pd.read_csv(file_path,
                        parse_dates=['Time'],
                        header=18
                        )        
        
