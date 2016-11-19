#!/usr/bin/octave


a = randn(82,2) + 5;
b = randn(82,2) - 5;
b1 = b(1:41, :);
b2 = b(42:82, :);


X = [b1;a;b2];
%plot(X(:,1), X(:,2), 'x');



a = -0:0.1:4;
b = -1.4*(a-2).^2 + 8;

a1 = a + 0.1*randn(size(a));
b1 = b + 0.1*randn(size(b));
a2 = a + 0.1*randn(size(a));
b2 = b + 0.1*randn(size(b));

a = [a1 a2];
b = [b1 b2];


x = -2:0.1:2;
y = x.^2;

x1 = x + 0.1*randn(size(x));
y1 = y + 0.1*randn(size(y));
x2 = x + 0.1*randn(size(x));
y2 = y + 0.1*randn(size(y));

x = [x1 x2];
y = [y1 y2];

Y = [x a;y b]';
%plot(Y(:,1), Y(:,2), 'x'); hold on;

noise = 20*(rand(164,3) - 0.5);
data = [Y, X, noise];
%plot(data(:,1), data(:,2), 'x'); hold on;
%plot(data(:,3), data(:,4), 'x'); hold on;


csvwrite('moon_164x7.csv', data, 'precision', 3)
