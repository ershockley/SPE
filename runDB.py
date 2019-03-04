import pymongo
import os

def get_collection():
    uri = 'mongodb://pax:%s@zenigata.uchicago.edu:27017/run'
    uri = uri % os.environ.get('MONGO_PASSWORD')
    c = pymongo.MongoClient(uri,
                            replicaSet='runs',
                            readPreference='secondaryPreferred')
    db = c['run']
    collection = db['runs_new']
    return collection

def get_did(run_id, detector='tpc'):
    collection = get_collection()
    
    if detector == 'tpc':
        query = {"detector" : "tpc",
                 "number" : int(run_id)}
    elif detector == 'muon_veto':
        query = {"detector" : "muon_veto",
                 "name" : run_id}
             

    cursor = collection.find(query, {"number" : True,
                                     "name" : True,
                                     "data" : True,
                                     "_id" : False})
    
    data = list(cursor)[0]['data']
    did = None
    
    for d in data:
        if d['host'] == 'rucio-catalogue' and d['status'] == 'transferred':
            did = d['location']

    return did

def get_name(run_id, detector='tpc'):
    if detector == 'muon_veto':
        return run_id
    else:
        collection = get_collection()
        cursor = collection.find({'detector' : 'tpc', 'number' : int(run_id)}, 
                                 {'name' : 1})

        return list(cursor)[0]['name']

def events_per_file(run_id, detector='tpc'):
    collection = get_collection()
    
    if detector == 'tpc':
        
        query = {"detector" : "tpc", "number" : int(run_id)}
    else:
        query = {"detector" : "muon_veto", "name" : run_id}
    
    ret = list(collection.find(query, {'reader' : 1}))[0]

    if 'Zip' in  ret['reader']['ini']['trigger_config_override']:
        events_per_file = ret['reader']['ini']['trigger_config_override']['Zip']['events_per_file']
    else:
        events_per_file = 1000
    return events_per_file


def runs_by_source(source, num_range=None):
    collection = get_collection()

    query = {'detector' : 'tpc',
             'source.type' : source
             }
    if num_range:
        query['$and'] = [{'number' : {'$gte' : num_range[0]}},
                         {'number' : {'$lte' : num_range[1]}}
                         ]

    ret = list(collection.find(query, {'number': 1}))
    return [run['number'] for run in ret]

