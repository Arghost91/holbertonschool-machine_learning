#!/usr/bin/env python3
"""
Script that prints the location of a specific user
"""
import requests
import sys
import time


if __name__ == '__main__':
    """
    * The user is passed as first argument of the script with the full API
    URL, example: ./2-user_location.py https://api.github.com/users/holbertonschool
    * If the user doesn’t exist, print Not found
    * If the status code is 403, print Reset in X min where X is the number
    of minutes from now and the value of X-Ratelimit-Reset
    """
    r = requests.get(sys.argv[1])
    r_req = r.status_code
    if r_req == 200:
        print(r.json()["location"])
    elif r_req == 403:
        ratelimit = int(r.headers["X-Ratelimit-Reset"])
        time = int(time.time())
        reset = (ratelimit - time) / 60
        print("Reset in {} min".format(reset))
    else:
        print("Not Found")
