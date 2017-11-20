from analyze import get_thresholds


good_list=get_thresholds(11541)
bad_list=get_thresholds(14093)
bad_thresh=[]

for i in range(0, len(good_list)):
    if good_list[i]!=bad_list[i]:
        bad_thresh.append(i)

if bad_thresh !=[]:
    print(bad_thresh)
else:
    print('All runs are equal')

