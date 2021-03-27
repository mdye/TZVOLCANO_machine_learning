# Dependencies
import urllib.request


class ChordsAPI:
    def __init__(self):
        self.domain				= 'tzvolcano.chordsrt.com'


    def download_olo_csv_file(self, instrument_id, start, end):
        domain				= 'tzvolcano.chordsrt.com'
        base_api_dir = 'api/v1/data/'
        url = f'http://{domain}/{base_api_dir}/{instrument_id}.csv?start={start}&end={end}'
        print(url)
        with urllib.request.urlopen(url) as f:
            data = f.read().decode('utf-8')

        return(data)

    # def get_csv_data(self, instrument_id, start, end):
        