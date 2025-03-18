# Below is the python code to order points with Dynamo DesignScript API

import clr
clr.AddReference('ProtoGeometry')
from Autodesk.DesignScript.Geometry import Line, Point

def distanceTo(pointOne, pointTwo) :

    line = Line.ByStartPointEndPoint(pointOne, pointTwo)
    return line.Length
    
def orderPoints(points) :

    if not points : return []

    ordered = [points[0]]
    remaining = points[1:]

    while remaining :

        last = ordered[-1]

        # Find the nearest unvisited point
        nearest = min(remaining, key=lambda p: distanceTo(p, last))
        ordered.append(nearest)
        remaining.remove(nearest)

    return ordered

points = IN[0]
OUT =  orderPoints(points)