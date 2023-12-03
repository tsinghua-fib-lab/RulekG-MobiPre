# Copyright (c) Facebook, Inc. and its affiliates.

import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import random
import numpy as np
import time
from collections import defaultdict
from random import sample
import setproctitle
import shutil
setproctitle.setproctitle("location-yuqh")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')
DATA_PATH='./data1/'

import pandas as pd
dat = pd.read_csv('./data/dataset_TSMC2014_NYC.csv')

fsq = dat.values[:,[0,1,7]]
cate = dat.values[:,[1,3]]

category = defaultdict(str)
for line in cate:
    loc,cat = line
    category[loc] = cat

def tid_time(ts):
    tm = time.strptime(ts, "%a %b %d %H:%M:%S +0000 %Y")
    if tm.tm_wday in [0, 1, 2, 3, 4]:
        t_hour = tm.tm_hour * 2 
    else:
        t_hour = (tm.tm_hour + 24) * 2
    if tm.tm_min < 30:
        tid = t_hour
    else:
        tid = t_hour + 1
    return str(tid)

def prepare_dataset(path, name):

    time_int = 30
    sort_users =  defaultdict(list)
    for line in fsq:
        user,poi,ts = line
        sort_users[user].append([user,poi,ts])
    sort_users1 = defaultdict(list)
    for user in list(sort_users.keys())[:700]:
        if len(sort_users[user])>150:
            user_times = np.array(sort_users[user])[:,2]
            sort_users1[user] = [sort_users[user][0]]
            for i in range(len(user_times)-1):
                if time.mktime(time.strptime(sort_users1[user][-1][2], "%a %b %d %H:%M:%S +0000 %Y"))+ time_int*60 < time.mktime(time.strptime(user_times[i+1], "%a %b %d %H:%M:%S +0000 %Y")):
                    sort_users1[user].append(sort_users[user][i+1])
    sort_users = defaultdict(list)            
    train2 = []
    test2 = []
    valid2 = []
    for user in sort_users1.keys():
        traces = sort_users1[user]
        for i in range(1,int((len(traces)-1)*0.7)+1):
            times = set()
            for j in range(i):
                if time.mktime(time.strptime(traces[j][2], "%a %b %d %H:%M:%S +0000 %Y"))+ 25*60*60 > time.mktime(time.strptime(traces[i][2], "%a %b %d %H:%M:%S +0000 %Y")):
                    times.add(traces[j][1])
                times.add(traces[i-1][1])    
            train2.append([user] + list(times)+[traces[i][1]])
        for i in range(int((len(traces)-1)*0.7)+1, int((len(traces)-1)*0.8)+1):
            times = set()
            for j in range(i):
                if time.mktime(time.strptime(traces[j][2], "%a %b %d %H:%M:%S +0000 %Y"))+ 25*60*60 > time.mktime(time.strptime(traces[i][2], "%a %b %d %H:%M:%S +0000 %Y")):
                    times.add(traces[j][1])
                times.add(traces[i-1][1])
            valid2.append([user] + list(times)+[traces[i][1]])
        for i in range(int((len(traces)-1)*0.8)+1,len(traces)):
            times = set()
            for j in range(i):
                if time.mktime(time.strptime(traces[j][2], "%a %b %d %H:%M:%S +0000 %Y"))+ 25*60*60 > time.mktime(time.strptime(traces[i][2], "%a %b %d %H:%M:%S +0000 %Y")):
                    times.add(traces[j][1])
                times.add(traces[i-1][1])
            test2.append([user] + list(times)+[traces[i][1]])
    
    for user in sort_users1.keys():
        for i in range(len(sort_users1[user])-1):    
            new_line = sort_users1[user][i] + sort_users1[user][i+1]
            sort_users[user].append(new_line)
   
    records = 0
    for k in sort_users.keys():
        records += len(sort_users[k])
    print('name:',name,"records: {}".format(records))
    train1 = []
    valid1 = []
    test1 = []
    for k in sort_users.keys():
        train1 += sort_users[k][0:int(0.7*len(sort_users[k]))]
        valid1 += sort_users[k][int(0.7*len(sort_users[k])):int(0.8*len(sort_users[k]))]
        test1 += sort_users[k][int(0.8*len(sort_users[k])):]

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    # write ent to id / rel to id
    for (dic, f) in zip([train1,valid1,test1, train2, valid2,test2], ['train1','valid1','test1','train2','valid2','test2']):
        ff = open(os.path.join(path, f), 'w+')
        for line in dic:
            ff.write(str(line)+"\n")
        ff.close()
    
    files = ['train1', 'valid1', 'test1']
    users, access, transfer,inverse, pois,timestamps, deltats, sim = set(),set(), set(), set(),set(), set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            _, origin, timestamp1, lhs, rhs, timestamp = eval(line)
            ts = tid_time(timestamp)
            ts1 = time.mktime(time.strptime(timestamp1, "%a %b %d %H:%M:%S +0000 %Y"))
            ts2 = time.mktime(time.strptime(timestamp, "%a %b %d %H:%M:%S +0000 %Y"))
            deltat =  int((ts2-ts1)//3600)
            users.add(lhs)
            timestamps.add(ts)
            deltats.add(deltat)
        to_read.close()
    similar = open('./data/similar_foursquare', 'r').read().strip().split('\n')
    for line in similar:
        user1,rel,user2 = line.split("\t")
        users.add(int(user1))
        users.add(int(user2))
    sim.add("similar")
    access.add('access')
    inverse.add('inverse')
    transfer.add('transfer')
    transfer_files = ['train2', 'valid2', 'test2']
    for ff in transfer_files:
        transfer_file = open(os.path.join(path, ff), 'r').read().split('\n')
        for line in transfer_file:
            if len(line)>=5:
                for i in range(1,len(eval(line))):
                    pois.add(eval(line)[i])

    users_to_id = {x: i for (i, x) in enumerate(sorted(users))}
    pois_to_id = {x: i for (i, x) in enumerate(sorted(pois))}
    access_to_id = {x: i for (i, x) in enumerate(sorted(access))}
    inverse_to_id = {x: i for (i, x) in enumerate(sorted(inverse))}
    transfer_to_id = {x: i for (i,x) in enumerate(sorted(transfer))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}
    sim_to_id = {x: i for (i, x) in enumerate(sorted(sim))}
    n_relations = len(access) + len(transfer)
    
    print("{} users, {} pois, {} relations over {} timestamps".format(len(users),len(pois),n_relations,len(timestamps)))
   
    if os.path.exists(os.path.join(DATA_PATH, name)):
        shutil.rmtree(os.path.join(DATA_PATH, name))
    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([users_to_id, pois_to_id, access_to_id, transfer_to_id, timestamps_to_id,inverse_to_id], ['user_id', 'poi_id', 'access_id', 'transfer_id', 'ts_id','inverse_to_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    confidence1 =defaultdict(set)
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            _, origin, timestamp1, lhs, rhs, timestamp = eval(line)
            ts = tid_time(timestamp)
            ots = tid_time(timestamp1)
            ts1 = time.mktime(time.strptime(timestamp1, "%a %b %d %H:%M:%S +0000 %Y"))
            ts2 = time.mktime(time.strptime(timestamp, "%a %b %d %H:%M:%S +0000 %Y"))
            deltat =  int((ts2-ts1)//3600)

            examples.append([users_to_id[lhs], timestamps_to_id[ts], access_to_id['access'],pois_to_id[rhs], timestamps_to_id[ots], inverse_to_id['inverse'],pois_to_id[origin],int(deltat),transfer_to_id['transfer']])
            if f == 'train':
                confidence1[(pois_to_id[origin],pois_to_id[rhs])].add(deltat)
          
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    for ff in transfer_files:
        transfer_file = open(os.path.join(path, ff), 'r').read().split('\n')
        transfer_id = []
        for line in transfer_file:
            if len(line)>=5:
                transfer_id.append([users_to_id[eval(line)[0]]] + [pois_to_id[i] for i in eval(line)[1:]])
        out = open(Path(DATA_PATH) / name / (ff + '.pickle'), 'wb')
        pickle.dump(transfer_id, out)
        out.close() 

    path1 = []
    for (origin_id,loc_id) in confidence1.keys():
        for deltat in confidence1[(origin_id,loc_id)]:
            path1.append([origin_id,loc_id,deltat])
    out = open(Path(DATA_PATH) / name / ('confidence1.pickle'), 'wb')
    pickle.dump(np.array(path1).astype('uint64'), out)
    out.close()

    number = [len(users), len(pois),len(timestamps),len(access), len(inverse),max(deltats), len(transfer)]
    out = open(Path(DATA_PATH) / name / 'numbers.pickle', 'wb')
    pickle.dump(np.array(number).astype('uint64'), out)
    out.close()
    
     # 访问同地点用户
    similars = []
    for line in similar:
        user1,rel,user2 = line.split("\t")
        similars.append([users_to_id[int(user1)],sim_to_id[rel],users_to_id[int(user2)]])
    out = open(Path(DATA_PATH) / name / ('similar.pickle'), 'wb')
    pickle.dump(np.array(similars).astype('uint64'), out)
    out.close()
    
if __name__ == "__main__":
    datasets = ['foursquare1']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise

