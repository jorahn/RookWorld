# Download & prepare Google BigBench Checkmate in One Benchmark

import requests
import os

URL = "https://github.com/google/BIG-bench/raw/main/bigbench/benchmark_tasks/checkmate_in_one/task.json"

def download_file(url, filename):
    with open(filename, "wb") as file:
        response = requests.get(url)
        file.write(response.content)

os.makedirs("bb-cio", exist_ok=True)
download_file(URL, "bb-cio/task.json")