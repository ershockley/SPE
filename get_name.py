import pymongo
import os
import sys

def get_name(number):
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
                                     "name" : True,
                                     "_id" : False})
    
    return list(cursor)[0]['name']

if __name__ == '__main__':
    print(get_name(int(sys.argv[1])))

