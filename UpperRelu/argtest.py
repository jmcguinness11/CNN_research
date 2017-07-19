import sys
if len(sys.argv) != 2:
	print("wrong")
	exit(1)

x = int(sys.argv[1])
if(x < 0):
	print("wrong")
	exit(1)
print x
