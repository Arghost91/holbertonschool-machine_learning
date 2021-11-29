#!/usr/bin/env python3
"""
Method that returns the list of ships that can hold a given number of passengers
"""
import requests


def availableShips(passengerCount):
    """
    * Prototype: def availableShips(passengerCount):
    * Don’t forget the pagination
    * If no ship available, return an empty list.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    r = requests.get(url)
    ships = []
    while(r.status_code == 200):
        json_result = r.json()["results"]
        for ship in json_result:
            passen = ship["passen"].replace(',', '')
            if int(passen) >= passengerCount:
                ships.append(ship["name"])
            elif passen == 'unknown' or passen == 'n/a':
                continue
        url = json["next"]
    return ships
