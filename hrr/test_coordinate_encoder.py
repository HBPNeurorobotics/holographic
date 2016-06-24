from hrr import HRR
HRR.stddev = 0.1
HRR.input_range = [0, 10]

A = 2
B = 8

Pa = [ (0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
#Pb = [ (0,4), (1,4), (2,4), (3,4), (4,4), (5,4)]
items = ["computer", "car", "table", "chair", "door", "book"]

m = None
for i in range(len(Pa)):
    a = HRR(Pa[i]) * items[i]
    #b = HRR(Pb[i]) * B
    if m is None:
        m = a
    else:
        m += a



probe = HRR((1,1))
print(m / probe)