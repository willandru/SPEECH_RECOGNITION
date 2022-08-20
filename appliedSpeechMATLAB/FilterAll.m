%transfer function h, used for pre-enphasis of speech.
h=[1, -0.9375];
y=filter(h,1,s);
soundsc(y)
plot(y)