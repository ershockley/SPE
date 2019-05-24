import pymongo
import os
import sys
import datetime

uri = 'mongodb://pax:%s@zenigata.uchicago.edu:27017/run'
uri = uri % os.environ.get('MONGO_PASSWORD')
c = pymongo.MongoClient(uri,
                        replicaSet='runs',
                        readPreference='secondaryPreferred')
db = c['run']
collection = db['runs_new']


TAGS = ['spe_bottom', 'spe_topbulk', 'spe_topring', 'gain_step4', 'gain_step0']

class SPECal:
    bottom = None
    topbulk = None
    topring = None
    blank = None


    def __init__(self, run, runlist_dir='runlists'):
        # look for other SPE runs close to this one
        self.runlist_dir = runlist_dir
        os.makedirs(runlist_dir, exist_ok=True)
        query = {'detector': 'tpc',
                 '$and': [{'number': {'$gt': run - 20}},
                          {'number': {'$lt': run + 20}},
                          {'tags.name': {'$in': TAGS}
                           },
                          {'tags.name': {'$not': {'$in': ['bad', 'messy', 'donotprocess']}}}
                        ],
                 'source.type': 'LED',
                 }

        cursor = collection.find(query, {'number': 1,
                                         'tags.name': 1,
                                         'trigger.events_built': 1}
                                 )
        cursor = list(cursor)

        events = {name: 0 for name in ['bottom', 'topbulk', 'topring', 'blank']}
        runs = {name: None for name in ['bottom', 'topbulk', 'topring', 'blank']}

        tags = ['']
        for r in cursor:
            tag = [t['name'] for t in r['tags'] if t['name'] in TAGS][0]

            nevents = r['trigger']['events_built']
            spe_type = self.translate(tag)
            if nevents > events[spe_type]:
                runs[spe_type] = r['number']

        for key, run in runs.items():
            setattr(self, key, run)


    @classmethod
    def translate(cls, tagname):
        if tagname == 'spe_bottom':
            return 'bottom'
        elif tagname == 'spe_topbulk':
            return 'topbulk'
        elif tagname == 'spe_topring':
            return 'topring'
        elif tagname in ['gain_step4', 'gain_step0']:
            return 'blank'

    @property
    def runlist_file(self):
        return os.path.join(self.runlist_dir, "runlist_%i_%i_%i.txt" % (self.bottom,
                                                                        self.topbulk,
                                                                        self.topring)
                            )

    def write(self):
        with open(self.runlist_file, 'w') as f:
            f.write("%i\n" % self.blank)
            f.write("%i\n" % self.bottom)
            f.write("%i\n" % self.topbulk)
            f.write("%i\n" % self.topring)

    @property
    def verified(self):
        return all([r is not None for r in [self.bottom, self.topbulk, self.topbulk, self.blank]])


    def print(self):
        print("---------------------")
        print("Bottom: %d" % self.bottom)
        print("TopBulk: %d" % self.topbulk)
        print("TopRing: %d" % self.topring)
        print("Blank: %d" % self.blank)
        print("---------------------")

def write_spe_lists(write=False):
    # get all spe_bottom runs
    query = {"detector":"tpc", 
             "source.type" : "LED",
             "number" : {"$gt" : 6731},
             "tags.name": 'spe_bottom'
            }
    
    cursor = collection.find(query, {"number" : True,
                                     "_id":False})
    cursor = list(cursor)

    for brun in cursor:
        cal = SPECal(brun['number'])

        if cal.verified:
            if write and not os.path.exists(cal.runlist_file):
                cal.write()
        else:
            print("Missing calibration near run %d" % brun)

        #cal.print()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        write = True
    else:
        write = False
    write_spe_lists(write)

