clear all
close all
clc

%%
numhoursperday = 24;
numdaysperweek = 7;
numweeksperyear = 52;
max_locations = 10;

%%
meshSize = numhoursperday * numdaysperweek * numweeksperyear;  
t_all = linspace(0,numweeksperyear-1,meshSize);

%%
mag = 0.5;
lambda = 2.75; % rate of change per given period, e.g. per months.
expon = 3;
short_freq = 2.75;
T_temp = t_all.*randsample(0.5:0.1:1,1); % T_temp = t_all.*rand;
rate_fh = @(t) (mag+lambda.*sin(t).^expon + sin(2*pi*short_freq*t).^expon);



%% Location
loc_ID = repmat([0:max_locations-1]',meshSize,1);

XY = [40.397555	-3.743121
40.384048	-3.723542
40.365236	-3.694193
40.380978	-3.671503
40.388431	-3.699452
40.416686	-3.711931
40.408923	-3.682114
40.395112	-3.714438
40.425457	-3.692114
40.428233	-3.649202];

X_ID = repmat(XY(:,1),meshSize,1);
Y_ID = repmat(XY(:,2),meshSize,1);


%% ROAD DIRECTION

% To or from center direction. Other labels could be used if necessary
Road_dir = {
'TO_CENTER';
'AWAY_FROM_CENTER';
'TO_CENTER';
'TO_CENTER';
'AWAY_FROM_CENTER';
'AWAY_FROM_CENTER';
'TO_CENTER';
'TO_CENTER';
'AWAY_FROM_CENTER';
'TO_CENTER';
};

Road_direction = repmat(Road_dir,meshSize,1);


%% NUMBER OF LANES

% Other number of lanes distribution could be used for each section
Numlanes = [1;2;4;1;2;4;2;2;1;1];
Number_of_lanes = repmat(Numlanes,meshSize,1);


%% Time
hours = repmat(repelem([0:numhoursperday-1]',max_locations) , numdaysperweek*numweeksperyear , 1);
days = repmat(repelem([1:numdaysperweek]',numhoursperday*max_locations) , numweeksperyear , 1);
weeks = repelem([1:numweeksperyear]',numhoursperday*numdaysperweek*max_locations);

%% PRECIPITATION

% mean precipitation rate in mm per hour
mean_precipitation_rate = rate_fh(T_temp)';
ifxneg = find(mean_precipitation_rate<0);
mean_precipitation_rate(ifxneg) = 0;

% assume the same mean precipation rate in all locations
a  = 2*repelem(mean_precipitation_rate,max_locations,1)/sqrt(pi);
k = random('uniform',1,3,length(a),1);
precipitation_rate = random('Weibull',a,k);

thresholdrain = 7;
id = find(loc_ID == 2);
for i=1:length(id)
    if (precipitation_rate(id(i))>thresholdrain)
        precipitation_rate(id(i))= precipitation_rate(id(i))+abs(random('Extreme Value',thresholdrain,1));
    end
end

id = find(loc_ID == 7);
for i=1:length(id)
    if (precipitation_rate(id(i))>thresholdrain)
        precipitation_rate(id(i))= precipitation_rate(id(i))+abs(random('Extreme Value',thresholdrain,1));
    end
end

% idf = randsample(length(precipitation_rate),85000);
% precipitation_rate(idf) = 0;

figure, plot(t_all,mean_precipitation_rate,'k-','LineWidth',2)
figure, histogram(precipitation_rate)
figure, plot(precipitation_rate)
figure, plot(loc_ID,precipitation_rate,'o')


%% CONTEXT
%  public holidays
% sporting events 
% school term dates
% Construction
% Other social event

Context = cell(length(loc_ID),1);
Context(:) = {'NORMAL'};

id7 = find(days == 3 & loc_ID == 5 & hours >= 17 & hours <= 23);
Context(id7) = {'SPORTS'};

id8 = find(days == 7 & (weeks == 27 | weeks == 37 | weeks == 43) & hours >= 14 & hours <= 23 & loc_ID >= 1 & loc_ID <= 4);
Context(id8) = {'CARNIVAL'};

id9 = find((days == 1 | days == 2 | days == 3) & (weeks == 17 | weeks == 30 | weeks == 40) & hours >= 6 & hours <= 18 & loc_ID == 3);
Context(id9) = {'CONSTRUCTION'};

id10 = find((days == 1 | days == 2 | days == 3) & (weeks == 17 | weeks == 30 | weeks == 40) & hours >= 6 & hours <= 18 & loc_ID == 9);
Context(id10) = {'CONSTRUCTION'};


id0 = find(weeks == 1);
id1 = find(weeks == 52);
id2 = find(weeks == 10);
id3 = find(weeks == 25);
id4 = find(weeks == 32);
id5 = find(weeks == 33);
id6 = find(weeks == 34);
ids = [id0;id1;id2;id3;id4;id5;id6];
Context(ids) = {'HOLIDAY'};

% ACCIDENTS
% Say 200 accidents occured all hours of a year
ida = randsample(length(loc_ID),200);
Context(ida) = {'ACCIDENT'};


%% TRAFFIC SPEED
avg_traffic_speed = 65; % km/h
traffic_speed = ones(length(loc_ID),1).*avg_traffic_speed.*random('normal',1,0.1,length(loc_ID),1);

% Effect of location
id = find(loc_ID == 2);
traffic_speed(id) = traffic_speed(id) + random('Extreme Value',35,8, length(id),1);

id = find(loc_ID == 5);
traffic_speed(id) = traffic_speed(id) + random('normal',18,2, length(id),1);

id = find(loc_ID == 6);
traffic_speed(id) = traffic_speed(id) + random('Extreme Value',20,3, length(id),1);

id = find(loc_ID == 9 & precipitation_rate>10);
traffic_speed(id) = traffic_speed(id) + random('Extreme Value',-25,4, length(id),1);

% Effect of context
idx = strcmp(Context,'CONSTRUCTION');
idx = find(idx==1);
traffic_speed(idx) = traffic_speed(idx)./random('uniform',5,10,length(idx),1); 

idx = strcmp(Context,'CARNIVAL');
idx = find(idx==1);
traffic_speed(idx) = traffic_speed(idx)./random('uniform',5,10,length(idx),1); 

idx = strcmp(Context,'HOLIDAYS');
idx = find(idx==1);
traffic_speed(idx) = traffic_speed(idx)./random('uniform',1,5,length(idx),1); 

idx = strcmp(Context,'SPORTS');
idx = find(idx==1);
traffic_speed(idx) = traffic_speed(idx)./random('uniform',1,2,length(idx),1); 

idx = strcmp(Context,'ACCIDENT');
idx = find(idx==1);
traffic_speed(idx) = traffic_speed(idx)./random('uniform',11,20,length(idx),1); 


% Effect of precipitation
thresholdrain1 = 5;
thresholdrain2 = 12;
idx = find(precipitation_rate>=thresholdrain1 & precipitation_rate<thresholdrain2);
traffic_speed(idx) = traffic_speed(idx)./random('uniform',1,2,length(idx),1); 
idx = find(precipitation_rate>=thresholdrain2);
traffic_speed(idx) = traffic_speed(idx)./random('uniform',5,10,length(idx),1); 

traffic_speed(traffic_speed<=5) = 5; 

figure, plot(loc_ID,traffic_speed,'o')
figure, plot(precipitation_rate,traffic_speed,'o')


%% TRAFFIC INTENSITY at Time t+60min:  cars per hour
avg_traffic_intensity = 2000; % cars/h/location over a week
traffic_intensity_plus_60min = ones(length(loc_ID),1).*avg_traffic_intensity.*random('normal',1,0.2,length(loc_ID),1);

id = find(loc_ID == 2 | loc_ID == 5 | loc_ID == 6);
traffic_intensity_plus_60min(id) = traffic_intensity_plus_60min(id).*1.75;

id = find(loc_ID == 9);
traffic_intensity_plus_60min(id) = traffic_intensity_plus_60min(id).*1.25;

figure, plot(loc_ID,traffic_intensity_plus_60min,'o')


% Intensity drops by half on the weekend
id = find(days >= 6 & days <= 7);
traffic_intensity_plus_60min(id) = traffic_intensity_plus_60min(id)./2;

% Normally higher at certain hours of the day
id = find(hours < 6);
traffic_intensity_plus_60min(id) = traffic_intensity_plus_60min(id)./2;
id = find(hours >= 21);
traffic_intensity_plus_60min(id) = traffic_intensity_plus_60min(id)./1.5;

id = find(hours > 10 & hours <= 16);
traffic_intensity_plus_60min(id) = traffic_intensity_plus_60min(id)./1.25;

figure, plot(days,traffic_intensity_plus_60min,'o')
figure, plot(hours,traffic_intensity_plus_60min,'o')

% Effect of context
idx = strcmp(Context,'CONSTRUCTION');
idx = find(idx==1);
traffic_intensity_plus_60min(idx) = traffic_intensity_plus_60min(idx)./random('uniform',5,10,length(idx),1); 

idx = strcmp(Context,'CARNIVAL');
idx = find(idx==1);
traffic_intensity_plus_60min(idx) = traffic_intensity_plus_60min(idx)./random('uniform',5,10,length(idx),1); 

idx = strcmp(Context,'HOLIDAYS');
idx = find(idx==1);
traffic_intensity_plus_60min(idx) = traffic_intensity_plus_60min(idx)./random('uniform',1,5,length(idx),1); 

idx = strcmp(Context,'SPORTS');
idx = find(idx==1);
traffic_intensity_plus_60min(idx) = traffic_intensity_plus_60min(idx)./random('uniform',1,2,length(idx),1); 

idx = strcmp(Context,'ACCIDENT');
idx = find(idx==1);
traffic_intensity_plus_60min(idx) = traffic_intensity_plus_60min(idx)./random('uniform',11,20,length(idx),1); 


% Effect of precipitation
traffic_intensity_plus_60min(traffic_intensity_plus_60min<=0) = 0; 
thresholdrain1 = 5;
thresholdrain2 = 12;
idx = find(precipitation_rate>=thresholdrain1 & precipitation_rate<thresholdrain2);
traffic_intensity_plus_60min(idx) = traffic_intensity_plus_60min(idx)./random('uniform',1,2,length(idx),1); 
idx = find(precipitation_rate>=thresholdrain2);
traffic_intensity_plus_60min(idx) = traffic_intensity_plus_60min(idx)./random('uniform',5,10,length(idx),1); 

traffic_intensity_plus_60min(traffic_intensity_plus_60min<=5) = 0; 
figure, plot(loc_ID,traffic_intensity_plus_60min,'o')
figure, plot(precipitation_rate,traffic_intensity_plus_60min,'o')
figure, plot(traffic_speed,traffic_intensity_plus_60min,'o')


%% TRAFFIC INTENSITY at Time - 60min:  cars per hour

traffic_intensity_minus_60min = traffic_intensity_plus_60min(1:end-max_locations);
traffic_intensity_minus_60min = [traffic_intensity_plus_60min(1:max_locations).*1.15;traffic_intensity_minus_60min];

%% TRAFFIC INTENSITY at Time - 120min:  cars per hour

traffic_intensity_minus_120min = traffic_intensity_minus_60min(1:end-max_locations);
traffic_intensity_minus_120min = [traffic_intensity_minus_60min(1:max_locations).*1.15;traffic_intensity_minus_120min];


%%
varNames = {'loc_ID' , 'X_ID' , 'Y_ID' , 'Number_of_lanes' , 'hours' , 'days' , 'Weeks' , 'precipitation_rate_mm'};
Xint = [loc_ID , X_ID , Y_ID , Number_of_lanes , hours , days , weeks , precipitation_rate];

OutGenTrafficSyntheticSamples = array2table(Xint,'VariableNames',varNames);

OutGenTrafficSyntheticSamples.Road_direction = Road_direction;

OutGenTrafficSyntheticSamples.Context = Context;
OutGenTrafficSyntheticSamples.traffic_speed = ceil(traffic_speed);
% OutGenTrafficSyntheticSamples.traffic_intensity_minus_120min = ceil(traffic_intensity_minus_120min);
% OutGenTrafficSyntheticSamples.traffic_intensity_minus_60min  = ceil(traffic_intensity_minus_60min);
OutGenTrafficSyntheticSamples.traffic_intensity_plus_60min   = ceil(traffic_intensity_plus_60min);

writetable(OutGenTrafficSyntheticSamples,'./Data/OutGenTrafficSyntheticSamples.csv') 
