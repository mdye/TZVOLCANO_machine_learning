import pandas as pd
import urllib.request

# Define parameters for data retrieval
INSTRUMENT_ID = '1'



def load_olo_data(instrument_id, start, end):
	# http://tzvolcano.chordsrt.com/api/v1/data/1.csv?start=2021-01-03T00:00&end=2021-01-04T00:00

    url = f'http://tzvolcano.chordsrt.com/api/v1/data/{instrument_id}.csv?start={start}&end={end}'
    print(url)

    
    return pd.read_csv(url,
#                     index_col='Time', 
                    parse_dates=['Time'],
                    header=18
                    )


def download_olo_csv_file(instrument_id, start, end):
	url = f'http://tzvolcano.chordsrt.com/api/v1/data/{instrument_id}.csv?start={start}&end={end}'
	print(url)
	with urllib.request.urlopen(url) as f:
		data = f.read().decode('utf-8')

	# print(html)
	return(data)

# original_data = load_olo_data(INSTRUMENT_ID, START, END)    


import pandas as pd
from datetime import datetime


# pd.date_range(end = datetime.today(), periods = 100).to_pydatetime().tolist()

#OR

start_dates = pd.date_range(start="2021-01-01",end="2021-03-20")
end_dates = start_dates + pd.DateOffset(days=1)

file_name = "tzvolcano_concatneaded_data.csv"

f = open(file_name, 'w')


for index, start_date in enumerate(start_dates):
	end_date = end_dates[index]

	# FORMAT: 2021-01-03T00:00
	start = start_date.strftime("%Y-%m-%dT00:00")
	end = end_date.strftime("%Y-%m-%dT00:00")

	print(index,start,end)

	data_str = download_olo_csv_file(INSTRUMENT_ID, start, end)
	lines = data_str.splitlines()

	# Write the header if this is the first download
	if index == 0:
	  header = "\n".join(lines[0:19])
	  f.write(header)
	  f.write("\n")

	# write the rest of the data
	f.write("\n".join(lines[20:len(lines)]))

	
f.write("\n")
f.close


