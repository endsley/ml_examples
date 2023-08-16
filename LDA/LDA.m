#!/usr/bin/octave

x = -4:4;


n0 = 50;
n1 = 50;

%	Generate data
set_0 = [randn(n0,2)];
set_1 = [randn(n1,2) + 8];
total_data = [set_0; set_1];
center_bias = mean(total_data);
set_0 = set_0 - center_bias;
set_1 = set_1 - center_bias; 

%	Calculate mean of each cluster and scatter
m0 = mean(set_0);
m1 = mean(set_1);

M0 = (set_0 - repmat(m0, n0, 1))
S0 = (M0'*M0)/n0

M1 = (set_1 - repmat(m1, n1, 1));
S1 = (M1'*M1)/n1;

S = S1 + S0;
S_inv = inv(S);

%	Calculate the weights
w = S_inv*(m0' - m1')




w_direction = w/norm(w);

%project_0 = (set_0.*repmat(w',n0,1))'
%project_1 = (set_1.*repmat(w',n1,1))'

project_0 = kron((set_0*w_direction)',w_direction);
project_1 = kron((set_1*w_direction)',w_direction);

%y = (0.5-w(1)*x)/w(2);
y1 = w(1)*x; 
y2 = w(2)*x; 
%y = w(2)*x/w(1);

plot(set_0(:,1), set_0(:,2), 'bo')
hold on;
plot(set_1(:,1), set_1(:,2), 'ro')
plot(y1,y2)

plot(project_0(1,:), project_0(2,:),'go')
plot(project_1(1,:), project_1(2,:),'go')

print('LDA.png')
input('press any key')
