import json
import numpy as np

labels = json.load(open('labels.json'))

for key, value in labels.iteritems():
	ptg = np.array(value[0])
	ptg[1]=ptg[1]*600/480.0
	ptd = np.array(value[2])
	ptd[1]=ptd[1]*600/480.0
	midpoint = (ptg+ptd)/2
	print(midpoint)
	vector = ptd-ptg
	vector = vector/np.linalg.norm(vector)
	print(vector)
	labels[key] = [midpoint[0]-27,midpoint[1],-vector[1],vector[0]]
	print(labels[key])
json.dump(labels,open('newlabels.json','w'))

#    images = list(labels.keys())
#    images.sort(key=lambda name: int(name.split('.jpg')[0]))
