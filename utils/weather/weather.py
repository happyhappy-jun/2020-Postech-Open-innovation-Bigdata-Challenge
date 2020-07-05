# Crawl Weather Data from data.go.kr

import requests
import json


class Weather:
    URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    def __init__(self, _api_key):
        self.api_key = _api_key

    def get_raw(self, num_of_rows=10, page_no=1, data_type="JSON", data_code="ASOS", date_code="HR", start_date=20100101,
                start_hour=1, end_date=20100101,
                end_hour=2, num_observatory=108, num_results=10):
        res = requests.get(self.URL, params={
            'serviceKey': self.api_key,
            'numOfRows': num_of_rows,
            'pageNo': page_no,
            'dataType': data_type,
            'dataCd': data_code,
            'dateCd': date_code,
            'startDt': start_date,
            'startHh': start_hour,
            'endDt': end_date,
            'endHh': end_hour,
            'stnIds': num_observatory,
            'schListCnt': num_results
        })

        return json.loads(res.text)

