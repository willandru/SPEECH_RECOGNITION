aro= audiorecorder(16000,16,1);
record(aro);
pause(3);
stop(aro);

play(aro);
%pause(3.2)
speech =getaudiodata(aro, 'double');
sound(speech,8500);
%soundsc(speech,24000); % Scales the vector 
%sound(speech/max(abs(speech)), 8000) % Scales the vector

plot([1:size(speech)]/16000, speech)