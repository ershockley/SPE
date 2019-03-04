import pymongo
import os
import sys

uri = 'mongodb://eb:%s@xenon1t-daq.lngs.infn.it:27017,copslx50.fysik.su.se:27017,zenigata.uchicago.edu:27017/run'
uri = uri % os.environ.get('MONGO_PASSWORD')
c = pymongo.MongoClient(uri,
                        replicaSet='runs',
                        readPreference='secondaryPreferred')
db = c['run']
collection = db['runs_new']

def write_spe_lists(write = False):
    query = {"detector":"tpc", 
             "source.type" : "LED",
#             "tags" : {"$exists" : True},
             "comments": {"$exists" : True},
             "$and" : [{"number" : {"$gt" : 0}}, {"number" : {"$lt" : 11400}}]
            }
    
    cursor = collection.find(query, {"number" : True,
                                     "data" : True,
                                     "comments" : True,
                                     "tags" : True,
                                     "trigger.events_built" : True,
                                     "_id":False})
    cursor = list(cursor)

    if not write:
        print("cursor has %d runs" % len(cursor))
        
    spe_runs = []
    spe_bottom = []
    spe_topbulk = []
    spe_topring = []
    spe_blank = []
    
    missing_runs = []


    # this is an absolute mess, but tries to figure out which runs are which configuration
    for run in cursor:
        if 'tags' in run:
            if any(["bad" in t["name"] for t in run["tags"]]):
                continue
        # make sure we're only considering the long LED runs 
        if "events_built" not in run["trigger"] or run["trigger"]["events_built"] < 100000:
            continue
        
        if any(["Gain step_4" in com["text"] or "SPE acceptance step_0" in com["text"] or "LED calibration step 4" in com["text"]
                or "LED calibration, step 4" in com["text"] or "LED Calibration step 4" in com["text"]
                or "LED Calibration: step 4" in com['text'] or "LED Gain Calibration step4" in com['text']
                or "http://wims.univ-savoie.fr/wims/" in com['text'] or "PMT_callibration_step_4" in com["text"]
                or "#gain_step4" in com['text']
                or "LED calibration: step 4" in com['text']
                for com in run["comments"]]):
            spe_blank.append(run["number"])
            
        if any(["SPE" in com["text"] for com in run["comments"]]):
                
            if any(["topbulk" in com["text"] or "step_2" in com["text"] or "step 2" in com["text"]  or "3.7, 3.7, 3.8" in com["text"] for com in run["comments"]]):
                spe_topbulk.append(run["number"])
                    
            if any(["topring" in com["text"] or "step_3" in com["text"] or "step 3" in com["text"] or "3.84, 3.84, 3.94" in com["text"] or "topouter" in com['text'] or "TopRing" in com['text'] for com in run["comments"]]):
                spe_topring.append(run["number"])
                    
            if any(["bottom" in com["text"] or "step_1" in com["text"] or "step 1" in com["text"] or "3.6, 3.6, 3.7" in com["text"] for com in run["comments"]]):
                spe_bottom.append(run["number"])
            
            spe_runs.append(run["number"])

    for L in [spe_blank, spe_bottom, spe_topbulk, spe_topring]:
        remove_list = []
        for run in L:
            for r in L:
                if r <= run:
                    continue
                if abs(r-run) < 10:
                    remove_list.append(run)
        remove_list = list(set(remove_list))
        for r in remove_list:
            L.remove(r)
        
        blank_remove = []
        for blank in spe_blank:
            if not any([abs(blank - LED) < 10 for LED in spe_bottom]):
                blank_remove.append(blank)
            
        for b in list(set(blank_remove)):
            spe_blank.remove(b)

    print("Number of runs and most recent run of each type")
    print("blank" , len(spe_blank), max(spe_blank))
    print("bottom", len(spe_bottom), max(spe_bottom))
    print("topbulk", len(spe_topbulk), max(spe_topbulk))
    print("topring", len(spe_topring), max(spe_topring))

    if not (len(spe_blank) == len(spe_bottom) == len(spe_topbulk) == len(spe_topring)):
        print("Something went wrong, number of runs are not equal")
        print("blank: " , spe_blank)
        print("bottom: ", spe_bottom)
        print("topbulk: ", spe_topbulk)
        print("topring: ", spe_topring)


    wrote = []
    for blank, bot, bulk, ring in zip(spe_blank, spe_bottom, spe_topbulk, spe_topring):
        if not all([abs(blank - run) < 10 for run in [bot, bulk, ring] ]):
            continue
        filename = "./runlists/runlist_%i_%i_%i.txt" % (bot, bulk, ring)
        if not os.path.exists(filename):
            if write:
                wrote.append(filename)
                with open(filename, "w") as f:
                    f.write("%i\n" %blank)
                    f.write("%i\n" %bot)
                    f.write("%i\n" %bulk)
                    f.write("%i\n" %ring)
            else:
                print("%d %d %d %d" % (blank, bot, bulk, ring))
            
    if write:
        return wrote

    for file, runlist in zip(['spe_blank.txt', 'spe_bottom.txt', 'spe_topbulk.txt', 'spe_topring.txt'],
                             [spe_blank, spe_bottom, spe_topbulk, spe_topring]):
        with open(file, "w") as f:
            for run in runlist:
                f.write("%d\n" % run)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("writing these runs to files")
        write = True
    else:
        print("Dry run. These runs would be downloaded and analyzed.")
        write = False
    write_spe_lists(write)


