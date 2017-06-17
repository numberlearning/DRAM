import os
import glob
import re
import shutil

def get_num(fname):
	part = fname.split("_")[-1]
	return int(part.split(".")[0])

def replace_num(fname, num):
	part = fname.split("_")[-1]
	after = ".".join(part.split(".")[1:])
	return "save_%d.%s" % (num, after)

first_files = glob.glob("first-run/*.ckpt*")
second_files = glob.glob("second-run/*.ckpt*")
attention_files = glob.glob("attention-run/*.ckpt*")

first_max = max(map(get_num, first_files))
attention_max = max(map(get_num, attention_files))
to_append = attention_max - first_max


print first_max, attention_max

for i, file in enumerate(first_files):
	dst = file.replace("first-run", "combined-run")
	shutil.copy(file, dst)
	if i % 100 == 0:
		print "FIRST %d out of %d" % (i, len(first_files))

for i in range(0, to_append, 1000):
	src = "second-run/save_%d.ckpt" % i
	dst = "combined-run/save_%d.ckpt" % (first_max + 1000 + i, )
	shutil.copy(src, dst)

	src = "second-run/save_%d.ckpt.meta" % i
	dst = "combined-run/save_%d.ckpt.meta" % (first_max + 1000 + i, )
	shutil.copy(src, dst)
	print i, to_append


# for i, file in enumerate(second_files):
# 	num = get_num(file)
# 	dst = "combined-run/" + replace_num(file, num + m)
# 	shutil.copy(file, dst)
# 	if i % 100 == 0:
# 		print "%d out of %d" % (i, len(second_files))

