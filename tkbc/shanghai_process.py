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
import shutil
import setproctitle
setproctitle.setproctitle("location-yuqh")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')
DATA_PATH='./data1/'

def tid_48(ts):
    tm = time.localtime(int(ts))
    if tm.tm_wday in [0, 1, 2, 3, 4]:
        tid = tm.tm_hour
    else:
        tid = tm.tm_hour + 24
    return str(tid)

   
def prepare_dataset(path, name):
    source_file = open('./data/shanghai.txt', 'r').read().strip().split('\n')
    
    sort_dat = []
    for line in source_file:
        sort_dat.append([line.split('\t')[0],line.split('\t')[1]]+eval(line.split('\t')[2])[1:])
    sort_users = {}
    for line in sort_dat:
        user,ts,poi,cate1,cate2,cate3 = line
        if user not in sort_users:
            sort_users[user]=[line[1:]]
        else:
            sort_users[user].append(line[1:])
    sort_users1 = {}
    for user in sort_users.keys():
        traces = np.array(sort_users[user])
        traces_user = traces[traces[:,0].argsort()]
        if len(sort_users[user])-1>50:
            sort_users1[user] = []
            for i in range(len(traces_user)-1):
                new_line = [user]+ traces_user[i].tolist()+traces_user[i+1].tolist()
                sort_users1[user].append(new_line)
    pois=set()
    sort_users={}
    for k in sort_users1.keys():
        traces = sort_users1[k]
        for i in range(len(traces)):
            pois.add(traces[i][2])
            pois.add(traces[i][7])
        if len(pois)>10:
            sort_users[k]= traces
    
    train_dat = []
    test_dat = []
    valid_dat = []
    for k in sort_users.keys():
        train_dat += sort_users[k][0:int(0.8*len(sort_users[k]))]
        valid_dat += sort_users[k][int(0.8*len(sort_users[k])):int(0.9*len(sort_users[k]))]
        test_dat += sort_users[k][int(0.9*len(sort_users[k])):]
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    # write ent to id / rel to id
    for (dic, f) in zip([train_dat, valid_dat, test_dat], ['train', 'valid', 'test']):
        ff = open(os.path.join(path, f), 'w+')
        for line in dic:
            ff.write(str(line)+"\n")
        ff.close()
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(timestamp)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    categories = []
    files = ['train', 'valid', 'test']
    entities, relations, pois,timestamps, cat1_rel,cates1, origins, detats, cat2_rel,cates2, cat3_rel,cates3, train_pois= set(), set(), set(), set(),set(), set(), set(),set(),set(), set(), set(), set(), set()
    for f in files:
        if f == 'train':
            file_path = os.path.join(path, f)
            to_read = open(file_path, 'r')
            for line in to_read.readlines():
                if len(line) >10: 
                    lhs, timestamp1, origin, ori_cate1, ori_cate2,ori_cate3, timestamp, rhs, cate1, cate2, cate3 = eval(line.strip())
                    train_pois.add(rhs)
                    if [rhs,cate1,cate2,cate3] not in categories:
                        categories.append([rhs,cate1,cate2,cate3])
                    
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, timestamp1, origin, ori_cate1, ori_cate2,ori_cate3, timestamp, rhs, cate1, cate2, cate3 = eval(line.strip())
            ts = tid_48(timestamp)
            detat =  int((int(timestamp)-int(timestamp1))//3600)
            entities.add(lhs)
            pois.add(rhs)
            relations.add('transfer')
            timestamps.add(ts)
            cat1_rel.add('cate1')
            cates1.add(cate1)
            cates1.add(ori_cate1)
            cat2_rel.add('cate2')
            cates2.add(cate2)
            cates2.add(ori_cate2)
            cat3_rel.add('cate3')
            cates3.add(cate3)
            cates3.add(ori_cate3)
            origins.add(origin)
            detats.add(detat)
        to_read.close()
    explore_pois = pois.difference(train_pois)
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    pois_to_id = {x: i+1 for (i, x) in enumerate(sorted(train_pois))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}
    catrel1_to_id = {x: i for (i, x) in enumerate(sorted(cat1_rel))}
    cates1_to_id= {x: i for (i, x) in enumerate(sorted(cates1))}
    catrel2_to_id = {x: i for (i, x) in enumerate(sorted(cat2_rel))}
    cates2_to_id= {x: i for (i, x) in enumerate(sorted(cates2))}
    catrel3_to_id = {x: i for (i, x) in enumerate(sorted(cat3_rel))}
    cates3_to_id= {x: i for (i, x) in enumerate(sorted(cates3))}
    origins_to_id = {x: i for (i, x) in enumerate(sorted(origins))}
    detats_to_id = {x: i for (i, x) in enumerate(sorted(detats))}
    
    print("{} entities,{} train_pois, {} pois, {} relations, {} category1 over {} timestamps".format(len(entities), len(train_pois),len(pois),len(relations), len(cates1), len(timestamps)))
    explore_to_id = pois_to_id.copy()
    for i in range(len(explore_pois)):
        explore_to_id[list(explore_pois)[i]] = len(pois_to_id)+i
    n_relations = len(relations)
    n_entities = len(entities)
    if os.path.exists(os.path.join(DATA_PATH, name)):
        shutil.rmtree(os.path.join(DATA_PATH, name))
    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, pois_to_id, relations_to_id, timestamps_to_id, explore_to_id, cates1_to_id, cates2_to_id,cates3_to_id,detats_to_id], ['ent_id', 'poi_id', 'rel_id', 'ts_id', 'explore_to_id', 'cate1_id', 'cate2_id', 'cate3_id','detats_to_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples_explore = []
        examples = []
        if f =='train':
            for line in to_read.readlines():
                lhs, timestamp1, origin, origin_cate1, origin_cate2,origin_cate3, timestamp, rhs, cate1, cate2, cate3 = eval(line.strip())
                ts = tid_48(timestamp)
                ots = tid_48(timestamp1)
                detat =  int((int(timestamp)-int(timestamp1))//3600)
                try:
                    examples.append([entities_to_id[lhs], relations_to_id['transfer'], pois_to_id[rhs], timestamps_to_id[ts], catrel1_to_id['cate1'],cates1_to_id[cate1], origins_to_id[origin], timestamps_to_id[ots], detats_to_id[detat], catrel2_to_id['cate2'],cates2_to_id[cate2], catrel3_to_id['cate3'],cates3_to_id[cate3], cates1_to_id[origin_cate1],cates2_to_id[origin_cate2], cates3_to_id[origin_cate3]])
                except ValueError:
                     continue
        else:
             for line in to_read.readlines():
                    lhs, timestamp1, origin, origin_cate1, origin_cate2,origin_cate3, timestamp, rhs, cate1, cate2, cate3  = eval(line.strip())
                    ts = tid_48(timestamp)
                    ots = tid_48(timestamp1)
                    detat =  int((int(timestamp)-int(timestamp1))//3600)
                    if rhs in pois_to_id:
                        examples.append([entities_to_id[lhs], relations_to_id['transfer'], explore_to_id[rhs], timestamps_to_id[ts], catrel1_to_id['cate1'],cates1_to_id[cate1], origins_to_id[origin], timestamps_to_id[ots], detats_to_id[detat], catrel2_to_id['cate2'],cates2_to_id[cate2], catrel3_to_id['cate3'],cates3_to_id[cate3], cates1_to_id[origin_cate1],cates2_to_id[origin_cate2], cates3_to_id[origin_cate3]])
                        examples_explore.append([entities_to_id[lhs], relations_to_id['transfer'], explore_to_id[rhs], timestamps_to_id[ts], catrel1_to_id['cate1'],cates1_to_id[cate1], origins_to_id[origin], timestamps_to_id[ots], detats_to_id[detat], catrel2_to_id['cate2'],cates2_to_id[cate2], catrel3_to_id['cate3'],cates3_to_id[cate3], cates1_to_id[origin_cate1],cates2_to_id[origin_cate2], cates3_to_id[origin_cate3]])
                    else:
                        examples_explore.append([entities_to_id[lhs], relations_to_id['transfer'], explore_to_id[rhs], timestamps_to_id[ts], catrel1_to_id['cate1'],cates1_to_id[cate1], origins_to_id[origin], timestamps_to_id[ots], detats_to_id[detat], catrel2_to_id['cate2'],cates2_to_id[cate2], catrel3_to_id['cate3'],cates3_to_id[cate3], cates1_to_id[origin_cate1],cates2_to_id[origin_cate2], cates3_to_id[origin_cate3]])
                       
            
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()
        if f in ['valid','test']:
            out = open(Path(DATA_PATH) / name / (f + '_explore.pickle'), 'wb')
            pickle.dump(np.array(examples_explore).astype('uint64'), out)
            out.close()
    categories_r = []
    for line in categories:
        rhs,cate1,cate2,cate3 = line
        new_line = [pois_to_id[rhs],catrel1_to_id['cate1'],catrel2_to_id['cate2'],catrel3_to_id['cate3'],cates1_to_id[cate1],cates2_to_id[cate2],cates3_to_id[cate3]]
        categories_r.append(new_line)
    out = open(Path(DATA_PATH) / name / ('categories.pickle'), 'wb')
    pickle.dump(np.array(categories_r).astype('uint64'), out)
    out.close()
    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts,catrel1, cate1, origin, ots, detat, catrel2, cate2,catrel3, cate3,origin_cate1, origin_cate2, origin_cate3 in examples:
            #to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts,catrel1, cate1, origin, ots, detat, catrel2, cate2,catrel3, cate3, origin_cate1, origin_cate2, origin_cate3)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()
    
if __name__ == "__main__":
    datasets = ['shanghai_30']
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

