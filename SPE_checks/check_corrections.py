from analyze import get_corrections

#pretty sure this script is useless
good_list=list(get_corrections(11535))
bad_list=list(get_corrections(14088))
bad_corr={}

for i in range(0, len(good_list)):
    if good_list[i]!=bad_list[i]:
        bad_corr[i]=good_list[i], bad_list[i]

if bad_corr!={}:
    print(bad_corr)
else:
    print('All corrections are the same')
