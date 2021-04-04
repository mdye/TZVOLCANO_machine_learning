# Jupyter widgets
# https://dataviz.shef.ac.uk/blog/16/06/2020/Jupyter-Widgets

# Good tutorial
# https://towardsdatascience.com/bring-your-jupyter-notebook-to-life-with-interactive-widgets-bc12e03f0916

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets
from datetime import datetime



class chords_gui:
    def __init__(self):
        self.start_datetime_default = datetime.fromisoformat('2020-02-01')
        self.end_datetime_default = datetime.fromisoformat('2020-02-01')

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

        
    def start_end_widgets(self):
        
        self.gui = widgets.HBox([self.start_date, self.end_date])

        
        gui_output = widgets.Output()
        display(self.gui, gui_output)