import numpy as np

from collections import defaultdict, namedtuple
from dataclasses import dataclass


@dataclass
class TileSet:
    def __init__(self):
        self.tiles = defaultdict(lambda: defaultdict(dict))

    def add(self, tile):
        self.tiles[tile.metric][tile.country][tile.population] = tile

    def fetch(self, metric: str, country: str = None, population: str = None):
        output = []
        for m, countries in self.tiles.items():
            for c, populations in countries.items():
                for p, tile in populations.items():
                    if (
                        (m == metric)
                        and ((country or c) == c)
                        and ((population or p) in p)
                    ):
                        output.append(tile)
        return output

    def levels(self, metric: str, country: str = None, population: str = None):
        x = self.fetch(metric, country, population)
        Levels = namedtuple("Levels", ["metrics", "countries", "populations"])
        return Levels(
            np.unique([i.metric for i in x]),
            np.unique([i.country for i in x]),
            np.unique([i.population for i in x]),
        )
