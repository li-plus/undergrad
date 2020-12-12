patient = [0.2 10.4 0.3 0.4 10.9 11.3 1.1 2.0 12.4 16.2 2.1 17.6 18.9 3.3 3.8 20.7 4.5 4.8 24.0 25.4 4.9 40.0 5.0 42.2 5.3 50.0 60.0 7.5 9.8 45.0];
normal = [0.2 5.4 0.3 5.7 0.4 5.8 0.7 7.5 1.2 8.7 1.5 8.8 1.5 9.1 1.9 10.3 2.0 15.6 2.4 16.1 2.5 16.5 2.8 16.7 3.6 20.0 4.8 20.7 4.8 33.0];

lillietest(patient)
lillietest(normal)

[muHat,sigmaHat,muCI,sigmaCI] = normfit(normal)
[muHat,sigmaHat,muCI,sigmaCI] = normfit(patient)

[h,p,ci,stats] = ttest2(patient, normal)

patient = patient(1:length(patient)-5);
[muHat,sigmaHat,muCI,sigmaCI] = normfit(patient)
[h,p,ci,stats] = ttest2(patient, normal)
