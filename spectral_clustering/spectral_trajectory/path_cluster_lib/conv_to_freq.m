
function fft_out = conv_to_freq(data, start_freq, end_freq)
	fft_dat = abs(fft(data));
	start_bin = floor(start_freq*size(fft_dat,1)) + 1;
	end_bin = floor(end_freq*size(fft_dat,1)/2);

	fft_out = fft_dat(start_bin:end_bin, :);
end
