REFIT: Electrical Load Measurements

THE FOLLOWING DATASET IS COMPLETELY RAW APART FROM ACCOUNTING FOR DAYLIGHT SAVINGS TIME (UK)

INFORMATION
Collection of this dataset was supported by the Engineering and Physical Sciences Research Council (EPSRC) via the project entitled Personalised Retrofit Decision Support Tools for UK Homes using Smart Home Technology (REFIT), under Grant Reference EP/K002368/1 to the University of Strathclyde. REFIT is a collaboration among the Universities of Strathclyde, Loughborough and East Anglia. The dataset includes data from 20 households from the Loughborough area over the period 2013 - 2014.  Key findings of the study are available at http://gtr.rcuk.ac.uk/projects?ref=EP%2FK002368%2F1.

LICENCING
This work is licensed under the Creative Commons Attribution 4.0 International Public License. See https://creativecommons.org/licenses/by/4.0/legalcode for further details.
Please cite the following paper if you use the dataset:

@inbook{278e1df91d22494f9be2adfca2559f92,
title = "A data management platform for personalised real-time energy feedback",
keywords = "smart homes, real-time energy, smart energy meter, energy consumption, Electrical engineering. Electronics Nuclear engineering, Electrical and Electronic Engineering",
author = "David Murray and Jing Liao and Lina Stankovic and Vladimir Stankovic and Richard Hauxwell-Baldwin and Charlie Wilson and Michael Coleman and Tom Kane and Steven Firth",
year = "2015",
booktitle = "Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting",
}

Each of the houses is labelled, House 1 - House 21 (skipping House 14), each house has 10 power sensors comprising a current clamp for the household aggregate and 9 Individual Appliance Monitors (IAMs). Only active power in Watts is collected at 8-second interval.
The subset of all appliances in a household that was monitored reflects the document from DECC of the largest consumers in UK households, https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/274778/9_Domestic_appliances__cooking_and_cooling_equipment.pdf

FILE FORMAT
The file format is csv and is laid out as follows;
DATETIME, UNIX TIMESTAMP (UCT), Aggregate, Appliance1, Appliance2, Appliance3, ... , Appliance9
Additionally data was only recorded when there was a change in load; this data has been filled with intermediate values where not available. The sensors are also not synchronised as our collection script polled every 6-8 seconds; the sensor may have updated anywhere in the last 6-8 seconds.
The file name _Part1 refers to the first iteration of the database where sensors that were not available were set to 0.
The file name _Part2 refers to the second iteration of the database where sensors that were not available were set to NaN to distinguish from 0.

MISSING DATA
During the course of the study there are a few periods of missing data (notably February 2014). Outages were due to a number of factors, including household internet failure, hardware failures, network routing issues, etc.

Household Information
House, Occupancy, Construction Year, Appliances Owned, Type, Size
1	,	2	,	1975-1980				, 35 , Detached			, 4 bed
2	,	4	,	-						, 15 , Semi-detached	, 3 bed
3	,	2	,	1988					, 27 , Detached			, 3 bed
4	,	2	,	1850-1899 				, 33 , Detached			, 4 bed
5	,	4	,	1878					, 44 , Mid-terrace		, 4 bed
6	,	2	,	2005					, 49 , Detached			, 4 bed
7	,	4	,	1965-1974				, 25 , Detached			, 3 bed
8	,	2	,	1966					, 35 , Detached			, 2 bed
9	,	2	,	1919-1944				, 24 , Detached			, 3 bed
10	,	4	,	1919-1944				, 31 , Detached			, 3 bed
11	,	1	,	1945-1964				, 25 , Detached			, 3 bed
12	,	3	,	1991-1995				, 26 , Detached			, 3 bed
13	,	4	,	post 2002				, 28 , Detached			, 4 bed
15	,	1	,	1965-1974				, 19 , Semi-detached	, 3 bed
16	,	6	,	1981-1990				, 48 , Detached			, 5 bed
17	,	3	,	mid 60s					, 22 , Detached			, 3 bed
18	,	2	,	1965-1974				, 34 , Detached			, 3 bed
19	,	4	,	1945-1964				, 26 , Semi-detached	, 3 bed
20	,	2	,	1965-1974				, 39 , Detached			, 3 bed
21	,	4	,	1981-1990				, 23 , Detached			, 3 bed

APPLIANCE LIST
The following list shows the appliances that were known to be monitored at the beginning of the study period. Although occupants were asked not to remove or switch appliances monitored by the IAMs, we cannot guarantee this to be the case. It should also be noted that Television and Computer Site may consist of multiple appliances, e.g. Television, SkyBox, DvD Player, Computer, Speakers, etc. Makes and Models specified here are gathered from pictures gathered by the installation team.

House 1
0.Aggregate
1.Fridge, Hotpoint, RLA50P
2.Freezer(1),Beko, CF393APW
3.Freezer(2), Unknown, Unknown
4.Washer Dryer, Creda, T522VW
5.Washing Machine, Beko, WMC6140
6.Dishwasher, Bosch, Unknown
7.Computer, Lenovo, H520s
8.Television Site, Toshiba, 32BL502b
9.Electric Heater, GLEN, 2172

House 2
0.Aggregate,
1.Fridge-Freezer, Unknown, Unknown
2.Washing Machine, LG, F1289TD
3.Dishwasher, Unknown, Unknown
4.Television Site,
5.Microwave, Unknown, Unknown
6.Toaster, Unknown, Unknown
7.Hi-Fi, Unknown, Unknown
8.Kettle, Unknown, Unknown
9.Overhead Fan

House 3
0.Aggregate,
1.Toaster, Dualit, DPP2
2.Fridge-Freezer, Whirlpool, ARC7612
3.Freezer, Frigidaire, Freezer Elite
4.Tumble Dryer, Unknown, Unknown
5.Dishwasher, Bosch, Exxcel Auto Option
6.Washing Machine, Unknown, Unknown
7.Television Site, Samsung, LE46A656A1FXXU
8.Microwave, Panasoinc, NN-CT565MBPQ
9.Kettle, Dualit, JKt3

House 4
0.Aggregate,
1.Fridge, Neff, K1514X0GB/31
2.Freezer, Ocean, UF 1025
3.Fridge-Freezer, Ariston, DF230
4.Washing Machine(1), Servis, 6065
5.Washing Machine(2), Zanussi, Z917
6.Desktop Computer, Unknown, Unknown
7.Television Site, Sony, KDL-32W706B
8.Microwave, Matsui, 170TC
9.Kettle, Swan, Unknown

House 5
0.Aggregate,
1.Fridge-Freezer, Fisher & Paykel, Unknown
2.Tumble Dryer, Unknown, Unknown
3.Washing Machine, AEG, L99695HWD
4.Dishwasher, Unknown, Unknown
5.Desktop Computer, Unknown, Unknown
6.Television Site, Unknown, Unknown
7.Microwave, Unknown, Unknown
8.Kettle, Logik, L17SKC14
9.Toaster, Breville, TT33

House 6
0.Aggregate,
1.Freezer, Whirlpool, CV128W
2.Washing Machine, Bosch, Classixx 1200 Express
3.Dishwasher, Neff, Unknown
4.MJY Computer, Unknown, Unknown
5.TV/Satellite, Samsung, UE55F6500SB
6.Microwave, Neff, H5642N0GB/02
7.Kettle, ASDA, GPK101W
8.Toaster, Breville, PT15
9.PGM Computer, Unknown, Unknown

House 7
0.Aggregate,
1.Fridge, Bosch, KSR30422GB
2.Freezer(1), Whirlpool, AFG 392/H
3.Freezer(2), Unknown, Unknown
4.Tumble Dryer, White Knight, Unknown
5.Washing Machine, Bosch, Unknown
6.Dishwasher, Unknown, Unknown
7.Television Site,
8.Toaster, Unknown, Unknown
9.Kettle, Sainsburys, 121988254

House 8
0.Aggregate,
1.Fridge, Liebherr, KP2620
2.Freezer, Unknown, Unknown
3.Washer Dryer, Zanussi, Unknown
4.Washing Machine,
5.Toaster, Bosch, TAT6101GB/02
6.Computer, Unknown, Unknown
7.Television Site, Sony, KDL-32V2000
8.Microwave, Panasoinc, NN-CT565MBPQ
9.Kettle, Morphy Richards, 43615

House 9
0.Aggregate,
1.Fridge-Freezer, Bosch, KGH34X05GB/05
2.Washer Dryer, Hotpoint, TCM580
3.Washing Machine, Bosch, Classixx 6 1200 Express
4.Dishwasher, Bosch, Classixx
5.Television Site, LG, 32LH3000
6.Microwave, Argos, MM717CFA
7.Kettle, Russel Hobbs, Unknown
8.Hi-Fi, Unknown, Unknown
9.Electric Heater

House 10
0.Aggregate,
1.Magimix(Blender), Unknown, Unknown
2.Toaster, Unknown, Unknown
3.Chest Freezer, Unknown, Unknown
4.Fridge-Freezer, Unknown, Unknown
5.Washing Machine, Beko, WI1382
6.Dishwasher, AEG, Unknown
7.Television Site, Samsung, UE40ES5500K
8.Microwave, Unknown, Unknown
9.K Mix, Unknown, Unknown

House 11
0.Aggregate,
1.Fridge, Gorenje, HPI 1566
2.Fridge-Freezer, Unknown, Unknown
3.Washing Machine, Unknown, Unknown
4.Dishwasher, Unknown, Unknown
5.Computer Site, Unknown, Unknown
6.Microwave, Unknown, Unknown
7.Kettle, Unknown, Unknown
8.Router, Unknown, Unknown
9.Hi-Fi, Unknown, Unknown

House 12
0.Aggregate,
1.Fridge-Freezer, Gorenje, HZS 3266
2.???, Unknown, Unknown
3.???, Unknown, Unknown
4.Computer Site, Unknown, Unknown
5.Microwave, Unknown, Unknown
6.Kettle, Unknown, Unknown
7.Toaster, Unknown, Unknown
8.Television, Unknown, Unknown
9.???, Unknown, Unknown

House 13
0.Aggregate,
1.Television Site, Samsung, UE55H6400AK
2.Freezer, Unknown, Unknown
3.Washing Machine, Unknown, Unknown
4.Dishwasher, Unknown, Unknown
5.???, Unknown, Unknown
6.Network Site, Unknown, Unknown
7.Microwave, Unknown, Unknown
8.Microwave, Unknown, Unknown
9.Kettle, Unknown, Unknown

House 15
0.Aggregate,
1.Fridge-Freezer, Unknown, Unknown
2.Tumble Dryer, Unknown, Unknown
3.Washing Machine, Beko, WMB91242LB
4.Dishwasher, Unknown, Unknown
5.Computer Site, Unknown, Unknown
6.Television Site, LG, 22LS4D
7.Microwave, Unknown, Unknown
8.Hi-Fi, Unknown, Unknown
9.Toaster, Unknown, Unknown

House 16
0.Aggregate,
1.Fridge-Freezer(1), Bosch, KGN30VW20G/01
2.Fridge-Freezer(2), Unknown, Unknown
3.Electric Heater(1), Unknown, Unknown
4.Electric Heater(2), Unknown, Unknown
5.Washing Machine, Bosch, WAB24262GB/01
6.Dishwasher, Unknown, Unknown
7.Computer Site, Unknown, Unknown
8.Television Site, Samsung, UE55HU8500T
9.Dehumidifier, Unknown, Unknown

House 17
0.Aggregate,
1.Freezer, Unknown, Unknown
2.Fridge-Freezer, Whirlpool, ARC 2990
3.Tumble Dryer, Unknown, Unknown
4.Washing Machine, Bosch, Exxcel 8 Vario Perfect
5.Computer Site, Unknown, Unknown
6.Television Site, Unknown, Unknown
7.Microwave, Matsui, M195T
8.Kettle, Russel Hobbs, 17869
9.TV Site(Bedroom), Unknown, Unknown

House 18
0.Aggregate,
1.Fridge(garage), LEC, R.403W
2.Freezer(garage), Unknown, Unknown
3.Fridge-Freezer, Unknown, Unknown
4.Washer Dryer(garage), Unknown, Unknown
5.Washing Machine, Unknown, Unknown
6.Dishwasher, Unknown, Unknown
7.Desktop Computer, Unknown, Unknown
8.Television Site, Unknown, Unknown
9.Microwave, Unknown, Unknown

House 19
0.Aggregate,
1.Fridge Freezer, Bosch, KGS-3272-GB/01
2.Washing Machine, Bosch, WAE24060GB/03
3.Television Site, Sony, KDL32EX703
4.Microwave, Kenwood, K20MSS10
5.Kettle, Breville, VKJ336
6.Toaster, Bellini, BET240
7.Bread-maker, Unknown, Unknown
8.Games Console, Unknown, Unknown
9.Hi-Fi, Unknown, Unknown

House 20
0.Aggregate,
1.Fridge, Unknown, Unknown
2.Freezer, Unknown, Unknown
3.Tumble Dryer, Unknown, Unknown
4.Washing Machine, Unknown, Unknown
5.Dishwasher, Unknown, Unknown
6.Computer Site, Unknown, Unknown
7.Television Site, Unknown, Unknown
8.Microwave, Unknown, Unknown
9.Kettle, Unknown, Unknown

House 21
0.Aggregate,
1.Fridge-Freezer, Samsung, SR-L3216B
2.Tumble Dryer, Unknown, Unknown
3.Washing Machine, Beko, WMB81241LW
4.Dishwasher, AEG, FAVORIT
5.Food Mixer, Unknown, Unknown
6.Television, Unknown, Unknown
7.Kettle, Unknown, Unknown
8.Vivarium, Unknown, Unknown
9.Pond Pump, Unknown, Unknown