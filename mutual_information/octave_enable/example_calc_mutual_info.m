#!/usr/bin/octave

x = [1;1;0;1;2;3;3];
y = [3;0;1;0;2;2;4];

%x = [0;0;1;1];
%y = [0;1;1;1];

[mInfo, xkey, ykey, x_prob, y_prob, x_y_prob] = calc_mutual_information(x,y);
mInfo
