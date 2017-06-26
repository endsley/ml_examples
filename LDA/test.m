#!/usr/bin/octave

x = 0:10;
y1 = 4*x + 4*x.^2;
y2 = x + 2*x.^2;

plot(y1, y2, 'ro')
input('press key')
