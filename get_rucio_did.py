import pymongo
import os
import sys

def get_did(number):
    uri = 'mongodb://pax:%s@zenigata.uchicago.edu:27017/run'
    uri = uri % os.environ.get('MONGO_PASSWORD')
    c = pymongo.MongoClient(uri,
                            replicaSet='runs',
                            readPreference='secondaryPreferred')
    db = c['run']
    collection = db['runs_new']
    
    query = {"detector" : "tpc",
             "number" : number}
             

    cursor = collection.find(query, {"number" : True,
                                     "data" : True,
                                     "_id" : False})
    
    data = list(cursor)[0]['data']
    did = None
    
    for d in data:
        if d['host'] == 'rucio-catalogue' and d['status'] == 'transferred':
            did = d['location']


    return did

if __name__ == '__main__':
    print(get_did(int(sys.argv[1])))

