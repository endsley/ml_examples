#!/usr/bin/octave


addpath('./path_cluster_lib')

c1 = randn(40,2);
c2 = 6*randn(100,2) + 10;
c3 = [c1;c2]';
distM = calc_Eucli_Distance_matrix(c3, 0);


%distM(:,1)
figure(1);hist(distM(:,3),30);
figure(2);hist(distM(:,4),30);
figure(3);hist(distM(:,45),30);
figure(4);hist(distM(:,66),30);

figure(5);
plot(c1(:,1), c1(:,2),'ro');
hold on;
plot(c2(:,1), c2(:,2),'o');
input('');
