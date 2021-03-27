
from ChordsAPI import ChordsAPI

domain = 'tzvolcano.chordsrt.com'
chords_api =  ChordsAPI(domain)



# Define parameters for data retrieval
INSTRUMENT_ID = '1'

start_str = "2021-01-01"
end_str = "2021-01-01"

chords_api.get_csv_data(INSTRUMENT_ID, start_str, end_str)
