import streamlit as st
import pandas as pd
import textwrap
from mpt_module import mpt_optimizer
# from mpt_optimizer import mpt_optimizer
# from mpt_module import mpt_optimizer

input_format = '''
    # Input format
    This tool expects your CSV files to have the following headers:
    | TICKER | DATE | CLOSE |
    | ------ | ---- | ----- |
    | Security ticker | Date | Closing price of this date |

    The data is expected to be comma-delimeted.
'''

upload_files_string = '''
    # Upload your files
'''

def main():

    st.write(textwrap.dedent(input_format))

    st.write(textwrap.dedent(upload_files_string))

    uploaded_files = st.file_uploader("Upload your CSV files with asset prices", accept_multiple_files=True)
    assets_dataframes = []

    # Initialize with values that will surely be overwritten
    begindate, enddate = (pd.Timestamp('1911-01-01'), pd.Timestamp('2026-01-01'))

    if len(uploaded_files) < 2:
        return
    
    for file in uploaded_files:
        try:
            df = pd.read_csv(filepath_or_buffer=file, delimiter=',', parse_dates=[1])
            df.columns = ['Ticker', 'Date', 'Close']
        except:
            st.error(
                f'Could not parse the following CSV file: {file.name}.\n\n'
                f'Make sure that the headers and delimiters are in the correct format.'
                )
        # Find a common timerange that will fit inside all the time series
        begindate = max(begindate, df['Date'].min())
        enddate = min(enddate, df['Date'].max())
        assets_dataframes.append(df)

    # If there are no common dates for all the assets
    if begindate > enddate - pd.Timedelta(days=10):
        st.error('No common timeframe could be chosen.')
        return

    st.write('# Construct the portfolio')
    d = st.date_input(
        label='Portfolio construction date',
        # Add beginning date + 10 days, because of rolling covariance of 10 days
        min_value=begindate + pd.Timedelta(days=10),
        max_value=enddate
        )
    required_return = st.number_input(label='What is the required return (as a decimal)?', placeholder='optional', value=None, step=0.01)

    date = f'{d.year}-{d.month}-{d.day}'
    try:
        fig = mpt_optimizer(assets_dataframes, date, 0 if required_return == None else required_return, 10)
        st.pyplot(fig)
    except:
        st.error('Failed to run the algorithm')

if __name__ == '__main__':
    main()