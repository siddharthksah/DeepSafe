import os, datetime
from distutils.dir_util import copy_tree
def save(url):
    mydir = os.path.join(os.getcwd(), "database", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    os.makedirs(mydir)
    copy_tree("./temp/", mydir)
    if url!="":
        with open(mydir + "/" + 'url.txt', 'a') as f:
            f.write(str(url))
            f.write("\n")