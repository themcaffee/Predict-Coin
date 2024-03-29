from datetime import timedelta, datetime
from time import sleep

import requests

from database import session, DataPoint, bulk_save_datapoints


class GDAX:
    def __init__(self, pair):
        self.pair = pair
        self.uri = 'https://api.gdax.com/products/{pair}/candles'.format(pair=self.pair)

    def fetch(self, start, end, granularity):
        # We will fetch the candle data in windows of maximum 100 items.
        delta = timedelta(minutes=granularity * 100)

        slice_start = start
        while slice_start != end:
            slice_end = min(slice_start + delta, end)
            slice_data = self.request_slice(slice_start, slice_end, granularity)
            slice_start = slice_end
            # If this data is not available, skip over it
            if slice_data:
                bulk_save_datapoints(slice_data, self.pair)
                session.commit()

    def request_slice(self, start, end, granularity):
        # Allow 3 retries (we might get rate limited).
        retries = 10
        for retry_count in range(0, retries):
            # From https://docs.gdax.com/#get-historic-rates the response is in the format:
            # [[time, low, high, open, close, volume], ...]
            response = requests.get(self.uri, {
                'start': GDAX.__date_to_iso8601(start),
                'end': GDAX.__date_to_iso8601(end),
                'granularity': granularity * 60  # GDAX API granularity is in seconds.
            })

            if response.status_code != 200 or not len(response.json()):
                if retry_count + 1 == retries:
                    print('Failed to get exchange data for ({}, {})!'.format(start, end))
                    return None
                else:
                    # Exponential back-off.
                    sleep(1.5 ** retry_count)
            else:
                # Sort the historic rates (in ascending order) based on the timestamp.
                result = sorted(response.json(), key=lambda x: x[0])
                return result

    @staticmethod
    def __date_to_iso8601(date):
        return '{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}'.format(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=date.hour,
            minute=date.minute,
            second=date.second)


if __name__ == '__main__':
    GDAX('ETH-USD').fetch(datetime(2017, 7, 1), datetime(2017, 12, 14), 1)
    datapoints = session.query(DataPoint).all()
    print("Data points added: {}".format(str(len(datapoints))))
