segment=s(4000:12000);
soundsc(segment);

[c, lags]= xcorr(segment);
subplot(2,1,1);
plot([0:1:8000]/8000, segment);
axis([0, 0.125, -0.05, 0.05]);
subplot(2,1,2);
plot(lags(1001:9001)/8000, c(1001:9001));
axis([0, 0.125, -0.2, 0.2])