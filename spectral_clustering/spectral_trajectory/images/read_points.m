#!/usr/bin/octave --silent


%load yoya_1.mat
%load yoya_1_b.mat
%load yoya_1_c.mat
%
%load yoya_2.mat
%load yoya_2_b.mat
%load yoya_2_c.mat
load points_3c.mat



data = points_3c;
data = [data(:,2), data(:,1)];
y1 = double(data - median(data));
yNorm = norm(y1,2,'rows');
yNorm = yNorm./max(yNorm);

b = y1(:,1) + i*y1(:,2);
a = angle(b);
a(a < 0) = 2*pi + a(a < 0);
[s,idx] = sort(a,'descend');

data(idx,:)
output = yNorm(idx)
save('./yoga_3c.mat','output')


%plot(yNorm(idx))
%figure(2);
%plot(abs(fft(yNorm(idx))))
%input('')
%[yNorm(idx) , idx]


%size(y1)
%norm(y1)

%norm(y1 ,2, 'rows')
%yoya_1_b - mean(yoya_1_b)
%yoya_1_c - mean(yoya_1_c)
