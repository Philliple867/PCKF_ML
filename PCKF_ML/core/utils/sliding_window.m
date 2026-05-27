function [Xseq, yseq] = sliding_window(Xn, yn, L)
% Overlapping L-length windows for LSTM input.
% Xn: [F x T], yn: [1 x T] -- both normalized.
N = size(Xn,2) - L;
Xseq = cell(N,1);
yseq = zeros(N,1);
for i = 1:N
    Xseq{i} = Xn(:, i:i+L-1);
    yseq(i)  = yn(i+L);
end
end
