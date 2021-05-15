# Dependencies
import urllib.request
import pandas as pd
import os

class ChordsAPI:
    def __init__(self, domain):
        self.domain         = domain
        self.base_api_dir   = 'api/v1/data/'
        self.local_data_dir = 'csv_files'

        os.makedirs(self.local_data_dir, exist_ok=True)
        


    def download_olo_csv_file(self, instrument_id, start, end):
        url = f'http://{self.domain}/{self.base_api_dir}/{instrument_id}.csv?start={start}&end={end}'
        
        print(url)

        with urllib.request.urlopen(url) as f:
            data = f.read().decode('utf-8')

        return(data)

    def get_file_name(self, instrument_id, start_str, end_str):
        readable_domain = self.domain.replace(".", "_")

        file_name = f'{self.local_data_dir}/{readable_domain}_instrument_id_{instrument_id}_{start_str}_to_{end_str}.csv'
        return(file_name)


    def get_csv_data(self, instrument_id, start_str, end_str):
        start_dates = pd.date_range(start=start_str, end=end_str)
        end_dates = start_dates + pd.DateOffset(days=1)

        file_name = self.get_file_name(instrument_id, start_str, end_str)

        f = open(file_name, 'w')
        
        print("Downloading data for instrument id ", instrument_id, " for dates from ", start_str, " to ", end_str)


        for index, start_date in enumerate(start_dates):
            end_date = end_dates[index]

            # FORMAT: 2021-01-03T00:00
            start = start_date.strftime("%Y-%m-%dT00:00")
            end = end_date.strftime("%Y-%m-%dT00:00")

            data_str = self.download_olo_csv_file(instrument_id, start, end)
            lines = data_str.splitlines()

            # Write the header if this is the first download
            if index == 0:
              header = "\n".join(lines[0:19])
              f.write(header + "\n")

            # write the rest of the data
            f.write("\n".join(lines[20:len(lines)]) + "\n")

            
        f.close
        
        print("Download complete, file created: ", file_name)