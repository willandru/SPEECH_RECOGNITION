%Applying the same filter but this time taking 240 sample frames

h=[1, -0.9375];
w=240;
n=floor(length(s)/w);
for k=1:n
    seg=s(1+(k-1)*w:k*w);
    segf=filter(h,1,seg);
    outsp(1+(k-1)*w:k*w)=segf;
end
soundsc(outsp)
plot(outsp)



