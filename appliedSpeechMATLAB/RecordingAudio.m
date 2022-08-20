aro= audiorecorder(8000,16,1);
record(aro);
pause(3.5);
stop(aro);
play(aro);
%pause(3.2)
s=getaudiodata(aro, 'double');
soundsc(s);
%soundsc(speech,24000); % Scales the vector 
%sound(speech/max(abs(speech)), 8000) % Scales the vector

plot([1:size(s)]/8000, s)