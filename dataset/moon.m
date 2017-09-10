#!/usr/bin/octave

N = 15000;		% half of samples
noise_N = 0;
total_d = 4 + noise_N;

a = randn(N,2) + 5;
b = randn(N,2) - 5;
b1 = b(1:(N/2), :);
b2 = b((N/2 + 1):N, :);


X = [b1;a;b2];
%plot(X(:,1), X(:,2), 'x');

a = linspace(0,4,N/2);

b = -1.4*(a-2).^2 + 8;

a1 = a + 0.1*randn(size(a));
b1 = b + 0.1*randn(size(b));
a2 = a + 0.1*randn(size(a));
b2 = b + 0.1*randn(size(b));

a = [a1 a2];
b = [b1 b2];


x = linspace(-2,2,N/2);
y = x.^2;

x1 = x + 0.1*randn(size(x));
y1 = y + 0.1*randn(size(y));
x2 = x + 0.1*randn(size(x));
y2 = y + 0.1*randn(size(y));

x = [x1 x2];
y = [y1 y2];

Y = [x a;y b]';
%plot(Y(:,1), Y(:,2), 'x'); hold on;

data = [Y, X];

if noise_N > 0		% only if there's noise
	noise = 20*(rand(N*2,noise_N) - 0.5);
	data = [data, noise];
end


%plot(data(:,1), data(:,2), 'x'); hold on;
%plot(data(:,3), data(:,4), 'x'); hold on;

original_label = [zeros(N/2,1); ones(N,1); zeros(N/2,1)];
alternative_label = [zeros(N,1); ones(N,1)];

%csvwrite('moon_164x7.csv', data, 'precision', 3)
csvwrite(['moon_' num2str(N*2) 'x' num2str(total_d) '.csv'], data, 'precision', 3)
csvwrite(['moon_' num2str(N*2) 'x' num2str(total_d) '_original_label.csv'], original_label, 'precision', 1)
csvwrite(['moon_' num2str(N*2) 'x' num2str(total_d) '_alt_label.csv'], alternative_label, 'precision', 1)
