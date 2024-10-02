# identity of people in balance_all dataset
import os
import pandas as pd

p_boundry = [17, 51, 70, 87, 120, 138, 157, 174, 191, 208, 227, 246, 265, 284, 303, 322, 341, 360, 379, 
398, 417, 436, 456, 475, 494, 513, 532, 551, 570, 589, 608, 627, 646, 665, 684, 703, 722, 741, 760, 779, 
798, 817, 855, 874, 912, 931, 950, 969, 1007, 1026, 1045, 1081, 1100, 1119, 1138, 1176, 1214, 1233, 1252, 1272]  # denotes the number of last image for each people 


data_dir = 'balance_all'
suffix = '.jpg'
order = [os.path.splitext(filename)[0] for filename in os.listdir(data_dir)
         if os.path.splitext(filename)[1] == suffix]
order.sort(key=lambda x: int(x.split('/')[-1]))  # NO.
label = []

for i in order:
	if int(i.split('/')[-1]) <= p_boundry[0]:
		label.append(0)
	else:
		for n, boundry in enumerate(p_boundry):
		    if int(i.split('/')[-1]) > boundry and int(i.split('/')[-1]) <= p_boundry[n+1]:
		        label.append(n+1)


label = pd.DataFrame(label)
label.to_csv(data_dir + '/label_identity.csv', encoding='utf-8')
# read






