import math
import scipy


def cylindrical_volume(height=None, radius=None, diameter=None):
    if height:
        if radius:
            return math.pi * radius**2 * height
        if diameter:
            return math.pi * (diameter/2)**2 * height
    else:
        return 0


def density(mass=None, volume=None):
    if volume != 0:
        return mass/volume
    else:
        return math.inf

# pass




