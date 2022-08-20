w=240;
hst=[];
n=floor(length(s)/w);
for k=1:n
    seg=s(1+(k-1)*w:k*w);
    [segf, hst]= filter(h,1,seg,hst);
    outsp2(1+(k-1)*w:k*w)=segf;
end
soundsc(outsp2)