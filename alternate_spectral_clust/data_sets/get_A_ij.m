#!/usr/bin/octave

function [ith, jth, A_ij] = get_A_ij(file_name, i,j)
	i = i + 1;
	j = j + 1;

	%m = csvread('./data_1.csv');
	m = csvread(file_name);

	ith = m(i,:);
	jth = m(j,:);

	differential = ith - jth;

	A_ij = differential(:)*differential;
end
