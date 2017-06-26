#!/usr/bin/octave

mets = [];
for m = 1:10
	met = nLDA();
	mets = [mets; met];
end

avg = mean(mets)
