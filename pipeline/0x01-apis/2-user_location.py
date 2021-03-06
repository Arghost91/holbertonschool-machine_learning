#!/usr/bin/env python3
"""
Script that prints the location of a specific user
"""
import requests
import sys
import time


if __name__ == '__main__':
    r = requests.get(sys.argv[1])
    r_req = r.status_code
    if r_req == 200:
        print(r.json()["location"])
    if r_req == 403:
        ratelimit = int(r.headers["X-Ratelimit-Reset"])
        time = int(time.time())
        reset = (ratelimit - time) / 60
        print("Reset in {} min".format(reset))
    if r_req == 404:
        print("Not Found")
