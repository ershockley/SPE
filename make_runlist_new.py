import pymongo
import os
import sys
import datetime

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
             "comments": {"$exists" : True},
             "number" : {"$gt" : 6731},
             "tags" : {"$exists" : True}
            }
    
    cursor = collection.find(query, {"number" : True,
                                     "data" : True,
                                     "comments" : True,
                                     "tags" : True,
                                     "trigger.events_built" : True,
                                     "_id":False})
    cursor = list(cursor)
    
    print(len(cursor))

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
        if any(["bad" in t["name"] for t in run["tags"]]):
            continue
        
        # make sure we're only considering the long LED runs 
        if "events_built" not in run["trigger"] or run["trigger"]["events_built"] < 100000:
            continue

        if run['tags'][0]['name'] == 'gain_step4':
                spe_blank.append(run["number"])
                            
        elif run['tags'][0]['name']=='spe_topbulk':
                spe_topbulk.append(run["number"])
                    
        elif run['tags'][0]['name']=='spe_topring':
                spe_topring.append(run["number"])
                    
        elif run['tags'][0]['name']=='spe_bottom':
                spe_bottom.append(run["number"])
            
        spe_runs.append(run["number"])
    
    
    
    #18193 was mislabeled?
    #spe_topring.append(18192)
    #spe_topbulk.remove(18192)
    #spe_topbulk.append(18191)
    
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
    for blank, bot, bulk, ring in zip(spe_blank, spe_bottom, spe_topbulk, sorted(spe_topring, reverse=True)):
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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("writing these runs to files")
        write = True
    else:
        print("Dry run. These runs would be downloaded and analyzed.")
        write = False
    write_spe_lists(write)

def get_dates(bottom_runs):
    query = {"detector": "tpc",
             "source.type": "LED",
             "comments": {"$exists": True},
             "number": {"$gt": 5038},
             "tags": {"$exists": True}
             }

    cursor = collection.find(query, {"number": True,
                                     "data": True,
                                     "comments": True,
                                     "tags": True,
                                     "trigger.events_built": True,
                                     "_id": False})
    cursor = list(cursor)

    datedict={}

    for run in cursor:
        if run["number"] in bottom_runs:
            datedict[run["number"]]=run["comments"][0]["date"]
            #rundate=datetime.datetime.strftime(run["tags"][0]["date"], '%x')
            #datedict[run["number"]]=datetime.datetime.strptime(rundate, '%x')

    return datedict
