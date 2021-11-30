#!/usr/bin/env python3
"""
Method that returns the list of names of the
home planets of all sentient species.
"""
import requests


def sentientPlanets():
    """
    * Prototype: def sentientPlanets():
    * Don’t forget the pagination
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    r = requests.get(url)
    planets = []
    while(r.status_code == 200):
        json_result = r.json()["results"]
        for specie in json_result:
            if specie["designation"] == "sentient" or specie["classification"] == "reptilian":
                if specie["homeworld"] is not None:
                    url_plan = specie["homeworld"]
                    planets.append(requests.get(url_plan).json()["name"])
        next = r.json()["next"]
        if(next is None):
            break
        r = requests.get(next)
    return planets
