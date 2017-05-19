# generate data for a confusion matrix
# Andrew Xia
# Dec 13 2016
# see mnist_to_spec, spec_to_mnist rtf files for data

# def minstToSpec():
# 	#define 10 as z, 0 as o
#
# 	#mnist to spec
# 	actualTotal = []
# 	outputTotal = []
#
# 	#generate actual labels
# 	for i in xrange(10):
# 		actualTotal.extend([i for j in xrange(100)])
#
# 	#0 case
# 	output = [0 for i in xrange(95)]
# 	output.extend([7,6,7,2,7])
# 	outputTotal.extend(output)
#
# 	#1 case
# 	output = [1 for i in xrange(74)]
# 	output.extend([7 for i in xrange(9)])
# 	output.extend([4 for i in xrange(8)])
# 	output.extend([9,6])
# 	output.extend([5 for i in xrange(5)])
# 	output.extend([3 for i in xrange(2)])
# 	outputTotal.extend(output)
#
# 	#2 case
# 	output = [2 for i in xrange(80)]
# 	output.extend([3 for i in xrange(6)])
# 	output.extend([5 for i in xrange(2)])
# 	output.extend([7 for i in xrange(6)])
# 	output.extend([6,8,8,1,0,0]) #one 0 is a 10
# 	outputTotal.extend(output)
#
# 	#3 case
# 	output = [3 for i in xrange(94)]
# 	output.extend([5,2,8,7,8,7]) #define 10 as 0
# 	outputTotal.extend(output)
#
# 	#4 case
# 	output = [4 for i in xrange(92)]
# 	output.extend([0,5,0,2,6,9,0,5]) #one 0 is a 10
# 	outputTotal.extend(output)
#
# 	#5 case
# 	output = [5 for i in xrange(98)]
# 	output.extend([9,1])
# 	outputTotal.extend(output)
#
# 	#6 case
# 	output = [6 for i in xrange(96)]
# 	output.extend([4,0,0,5]) #two 0 is a 10
# 	outputTotal.extend(output)
#
# 	#7 case
# 	output = [7 for i in xrange(97)]
# 	output.extend([5,5,0])
# 	outputTotal.extend(output)
#
# 	#8 case
# 	output = [8 for i in xrange(88)]
# 	output.extend([5 for i in xrange(4)])
# 	output.extend([3 for i in xrange(5)])
# 	output.extend([1,6,2])
# 	outputTotal.extend(output)
#
# 	#9 case
# 	output = [9 for i in xrange(71)]
# 	output.extend([7 for i in xrange(8)])
# 	output.extend([5 for i in xrange(16)])
# 	output.extend([3,6,6,4,8])
# 	outputTotal.extend(output)
#
# 	#labels
# 	labels = [i for i in xrange(10)]
#
# 	print len(actualTotal)
# 	print len(outputTotal)
# 	return outputTotal,actualTotal, labels


def specToMnist():

	#mnist to spec
	actualTotal = []
	outputTotal = []

	#generate actual labels
	for i in xrange(11):
		actualTotal.extend([i for j in xrange(163)])

	#z case - 5
	output = [0 for i in xrange(136)]
	output.extend([1 for i in xrange(9)])
	output.extend([6 for i in xrange(10)])
	output.extend([7,7,7,4,4,8,9,3])
	outputTotal.extend(output)

	#0 case - 11
	output = [0 for i in xrange(115)]
	output.extend([1 for i in xrange(7)])
	output.extend([4 for i in xrange(15)])
	output.extend([5 for i in xrange(6)])
	output.extend([6 for i in xrange(7)])
	output.extend([7 for i in xrange(6)])
	output.extend([8 for i in xrange(7)])
	outputTotal.extend(output)

	#1 case - 6
	output = [1 for i in xrange(157)]
	output.extend([4,4,4])
	output.extend([8,9,8])
	outputTotal.extend(output)

	#2 case - 13
	output = [2 for i in xrange(134)]
	output.extend([8 for i in xrange(14)])
	output.extend([0 for i in xrange(8)])
	output.extend([6,6,6,7,4,4,4])
	outputTotal.extend(output)

	#3 case - 8
	output = [3 for i in xrange(135)]
	output.extend([8 for i in xrange(17)])
	output.extend([7 for i in xrange(5)])
	output.extend([0,4,4,5,6,6])
	outputTotal.extend(output)

	#4 case - 2
	output = [4 for i in xrange(158)]
	output.extend([0,0,1]) #define 10 as 0
	output.extend([9,9]) #define 10 as 0
	outputTotal.extend(output)

	#5 case - 7
	output = [5 for i in xrange(145)]
	output.extend([6 for i in xrange(5)])
	output.extend([7 for i in xrange(5)])
	output.extend([0,3,3,3,4,8,8,9])
	outputTotal.extend(output)

	#6 case - 4
	output = [6 for i in xrange(156)]
	output.extend([0,0,5]) #two 0 is a 10
	output.extend([4,8,2,3])
	outputTotal.extend(output)

	#7 case - 9
	output = [7 for i in xrange(148)]
	output.extend([4 for i in xrange(3)])
	output.extend([6 for i in xrange(3)])
	output.extend([9,9,2,5,2,1,8,0,1])
	outputTotal.extend(output)

	#8 case - 6
	output = [8 for i in xrange(152)]
	output.extend([0,0,0,0])
	output.extend([6,3,6,6,6,3,5])
	outputTotal.extend(output)

	#9 case - 7
	output = [9 for i in xrange(128)]
	output.extend([4 for i in xrange(7)])
	output.extend([1 for i in xrange(8)])
	output.extend([7 for i in xrange(7)])
	output.extend([0,3,0,6,8,3,5,8,2,3,0,8,5])
	outputTotal.extend(output)

	#labels
	labels = [i for i in xrange(10)]

	print "Actuals",len(actualTotal)
	print "Output",len(outputTotal)
	return outputTotal, actualTotal, labels
