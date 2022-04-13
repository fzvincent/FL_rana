from var import *
exc=pd.read_excel("Number of selected clients over 1000 iterations.xlsx",header=None)
arr= exc.to_numpy()
arr=np.transpose(arr)
arr=arr-1
selection=[]
for i in range(1000):
    sele1iter=[]
    for j in arr[i]:
        if j!=-1:
            sele1iter.append(j)
    selection.append(sele1iter)
seleGreen=selection
print(1)

exc=pd.read_excel("LEARN.xlsx",header=None)
arr= exc.to_numpy()
arr=np.transpose(arr)
arr=arr-1
selection=[]
for i in range(1000):
    sele1iter=[]
    for j in arr[i]:
        if j!=-1:
            sele1iter.append(j)
    selection.append(sele1iter)
seleLearn=selection
print(1)