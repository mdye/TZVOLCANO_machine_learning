import pandas as pd

import urllib.request
from datetime import datetime


from ChordsAPI import ChordsAPI

chords_api =  ChordsAPI()



# Define parameters for data retrieval
INSTRUMENT_ID = '1'

start_str = "2021-01-01"
end_str = "2021-01-03"

file_name = f'tzvolcano_data_instrument_id_{INSTRUMENT_ID}_{start_str}_to_{end_str}.csv'




# def download_olo_csv_file(instrument_id, start, end):
# 	url = f'http://tzvolcano.chordsrt.com/api/v1/data/{instrument_id}.csv?start={start}&end={end}'
# 	print(url)
# 	with urllib.request.urlopen(url) as f:
# 		data = f.read().decode('utf-8')

# 	# print(html)
# 	return(data)




start_dates = pd.date_range(start=start_str, end=end_str)
end_dates = start_dates + pd.DateOffset(days=1)


f = open(file_name, 'w')


for index, start_date in enumerate(start_dates):
	end_date = end_dates[index]

	# FORMAT: 2021-01-03T00:00
	start = start_date.strftime("%Y-%m-%dT00:00")
	end = end_date.strftime("%Y-%m-%dT00:00")

	print(index,start,end)

	data_str = chords_api.download_olo_csv_file(INSTRUMENT_ID, start, end)
	lines = data_str.splitlines()

	# Write the header if this is the first download
	if index == 0:
	  header = "\n".join(lines[0:19])
	  f.write(header + "\n")

	# write the rest of the data
	f.write("\n".join(lines[20:len(lines)]) + "\n")

	
f.close


