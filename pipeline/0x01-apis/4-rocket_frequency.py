#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket
"""
import requests


if __name__ == '__main__':
    r = requests.get("https://api.spacexdata.com/v4/launches/")
    rockets = {}
    r_get = r.json()
    for launch in r_get:
        rocket = launch["rocket"]
        rocket_url = "https://api.spacexdata.com/v4/rockets/" + rocket
        r_rocket = requests.get(rocket_url)
        r_rocket_get = r_rocket.json()
        rocket_name = r_rocket_get["name"]
        if rocket_name in rockets.keys():
            rockets[rocket] = rockets[rocket] + 1
        else:
            rockets[rocket] = 1
    sorting = sorted(rockets.items(), key=lambda x: x[0])
    sorting = sorted(sorting, key=lambda x: x[1], reverse=True)
    for i in sorting:
        print("{}: {}".format(i[0], i[1]))
