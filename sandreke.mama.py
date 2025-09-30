from turtle import *
from math import *

speed(0)
bgcolor('black')
goto(0, -40)

for i in range(16):
    for j in range(18):
        color('#FFFFFF')
        right(90)
        circle(150 - j * 6, 90)
        left(90)
        circle(150 - j * 6, 90)
        right(180)
    circle(40, 24)

color('black')
shape('circle')
shapesize(0.5)
fillcolor('#FFA216')
golden_ang = 137.508
phi = golden_ang * (pi / 180)

for i in range(140):
    r = 3 * sqrt(i)
    theta = i * phi
    x = r * cos(theta)
    y = r * sin(theta)
    penup()
    goto(x, y)
    setheading(i * golden_ang)
    pendown()
    stamp()

done()