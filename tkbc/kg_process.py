import time
import os
import shutil
# "%Y%m%d%H%M"
def date_style_transformation(date,format1 = "%a %b %d %H:%M:%S +0000 %Y", format2 = "%Y-%m-%d"):
    time_array = time.strptime(date,format1)
    str_date = time.strftime(format2,time_array)
    return str_date

files = ['train1','valid1','test1']
path = './data/foursquare1'
if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)
result1 = []
result2 = []
result3 = []
path1 = './data/foursquare/'
dat = open(path1 + 'train1').read().strip().split('\n')
for line in dat:
    user,poi1,tim1,poi,poi,tim = eval(line)
    result1.append([user,poi1,date_style_transformation(tim1),user,poi,date_style_transformation(tim),'r_v','r_t'])
dat = open(path1 + 'valid1').read().strip().split('\n')
for line in dat:
    user,poi1,tim1,poi,poi,tim = eval(line)
    result2.append([user,poi1,date_style_transformation(tim1),user,poi,date_style_transformation(tim),'r_v','r_t'])
dat = open(path1 + '/test1').read().strip().split('\n')
for line in dat:
    user,poi1,tim1,poi,poi,tim = eval(line)
    result3.append([user,poi1,date_style_transformation(tim1),user,poi,date_style_transformation(tim),'r_v','r_t'])

for (dic, f) in zip([result1,result2,result3], ['train.txt','valid.txt','test.txt']):
    ff = open(os.path.join(path, f), 'w+')
    for line in dic:
        ff.write(str(line)+"\n")
    ff.close()

