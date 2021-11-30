#!/usr/bin/env python3
"""
Script that displays the upcoming launch with these information
"""
import requests


if __name__ == '__main__':
    r = requests.get("https://api.spacexdata.com/v4/launches/upcoming")
    r_get = r.json()
    dates = [i['date_unix'] for i in r_get]
    index = dates.index(min(dates))
    next = r_get[index]
    name = next["name"]
    date_local = next["date_local"]
    rocket = next["rocket"]
    rocket_url = "https://api.spacexdata.com/v4/rockets" + rocket
    r_rocket = requests.get(rocket_url)
    r_rocket_get = r_rocket.json()
    rocket_name = r_rocket_get["name"]
    launchpad = next["launchpad"]
    launchpad_url = "https://api.spacexdata.com/v4/launchpads" + launchpad
    r_launchpad = requests.get(launchpad_url)
    r_launchpad_get = r_launchpad.json()
    launchpad_name = r_launchpad_get["name"]
    launchpad_locality = r_launchpad_get["locality"]
print("{} ({}) {} - {} ({})".format(name,
                                   date_local,
                                   rocket_name,
                                   launchpad_name,
                                   launchpad_locality))
