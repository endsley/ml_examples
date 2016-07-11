#!/usr/bin/octave


%w = [ 0.3488;-0.0394;-0.905 ]
%w = [0.3595;-0.0396;-0.9323];
%w = [0.349;-0.039;-0.905];
%w = [1;0;0];

x = [0,0,0; 0,1,0; 2,0,1; 3,0,1]

U = [[-0.5042,  0.4782],
       [-0.4958,  0.5147],
       [-0.5143, -0.4297],
       [-0.4853, -0.5673]]

Y = [0 1;0 1;1 0;1 0]
W = [1 0;0 1;0 0]
N = size(x,1)
K = zeros(N,N);
Gamma = zeros(N,N);
sigma = 1;
Ky = Y*Y';
H = eye(N)- (1/N)*ones(N,N);
y_tilde = H*Ky*H;

for i = 1:N
	for j = 1:N
		xi = x(i,:);
		xj = x(j,:);
		xdif = xi - xj;

		K(i,j) = exp((-xdif*W*W'*xdif')/(2*sigma^2));
	end
end


D = 1./sqrt(sum(K))
for i = 1:N
	for j = 1:N
	
		ui = U(i,:);
		uj = U(j,:);
		part1 = (ui*uj')*D(i)*D(j);
		Gamma(i,j) = (ui*uj')*D(i)*D(j) - y_tilde(i,j);
	end
end

magnitude = sum(sum(Gamma.*K))













%R = zeros(4,4);
%for i = 1:4
%	for j = 1:4
%		[A,B, A_ij] = get_A_ij('./data_2.csv', i-1,j-1);
%		R(i,j) = g(i,j)*exp(-w'*A_ij*w/2);
%	end
%end
%
%sum(sum(R))


%R = zeros(4,4);
%last_mag = 1;
%
%for a = -2:0.01:4
%	for b = -2:0.01:4
%		for c = -2:0.01:4
%			w = [a;b;c];
%			w = w/norm(w);
%
%			
%			for i = 1:4
%				for j = 1:4
%					[A,B, A_ij] = get_A_ij('./data_2.csv', i-1,j-1);
%					R(i,j) = g(i,j)*exp(-w'*A_ij*w/2);
%				end
%			end
%			
%			
%			mag = sum(sum(R));
%			if mag > last_mag
%				last_mag = mag;
%
%				w
%				last_mag
%			end
%
%		end
%	end
%end
%
%
%
%
