#排序算法系列
#1选择排序 比较时间复杂度O（（n-1）n/2） 交换最坏时间复杂度 O（n-1）
def selection(a):
    for i in range(len(a)):
        m=i
        for j in range(i+1,len(a)):
            if a[m]>a[j]:
                m=j
        a[i],a[m]=a[m],a[i]
    return a
        
#2直接插入排序
def straightInsertionSort(a):
    for i in range(1,len(a)):
        for j in range(0,i):
            if a[j]<=a[i] and a[j+1]>=a[i]:
                temp=a[i]
                for k in range(i,j,-1):
                    a[k]=a[k-1]
                a[j+1]=temp
            if a[i]<a[0]:
                temp=a[i]
                for k in range(i,0,-1):
                    a[k]=a[k-1]
                a[0]=temp                 
    return a           
#shell排序
def shellsort(a):
    h=1
    while h<len(a)/3:
        h=3*h+1
    while h>=1:
        for i in range(h,len(a)):
            for j in range(0,i,h):
                if a[j]<=a[i] and a[j+1]>=a[i]:
                    temp=a[i]
                    for k in range(i,j,-1):
                        a[k]=a[k-1]
                    a[j+1]=temp
                if a[i]<a[0]:
                    temp=a[i]
                    for k in range(i,0,-1):
                        a[k]=a[k-1]
                    a[0]=temp    
        h=int((h-1)/3)
    return a
 #归并排序???????????????????????????
def mergeSort(a):   
    b=merge_sortRecursive(a)
    return b
def merge_sortRecursive(a):
    if len(a)<=1:
        return a;
    mid=int(len(a)/2)
    left=merge_sortRecursive(a[:mid])
    right=merge_sortRecursive(a[mid:])
    return merge(left,right)
def merge(a,b):      
    arr=[]
    i=j=0   
    while i<len(a) and j<len(b):
        if a[i]<b[j]:
            arr.append(a[i])
            i+=1                       
        else:
            arr.append(b[j])
            j+=1
                      
    while i<len(a):
        arr.append(a[i])     
        i+=1
    while j<len(b):
        arr.append(b[j]) 
        j+=1
    
    return arr
    
#快速排序
def partition(a,left,right):
    pivotValue=a[right-1]    
    storeIndex=left    
    for i in range(left,right-1):
        if a[i]<pivotValue:
            a[storeIndex],a[i]=a[i],a[storeIndex]
            storeIndex+=1
    a[storeIndex],a[right-1]=a[right-1],a[storeIndex]
    return storeIndex
def quicksort(a,left,right):   
    if right>left:
        pivotIndex=partition(a,left,right)
        quicksort(a,left,pivotIndex)
        quicksort(a,pivotIndex+1,right)
    return a    
#堆排序??????????????????????????????????????????




#堆排序??????????????????????????????????????????

#计数排序
def bucketSort(a):
    m=min(a)
    M=max(a)
    result=[]
    
    T=True
    if m<0:
        for i in range(len(a)):
            a[i]=a[i]-m
        T=False
        arr=[0]*(M-m+1)
    else:
        arr=[0]*(M+1)
    for i in range(len(a)):
        arr[a[i]]+=1
    for i in range(len(arr)):
        if arr[i]!=0:
            for j in range(arr[i]):
                if T==False:
                    result.append(i+m)
                else:
                    result.append(i)
    return result
 

 #基数排序
def fill(a):
    M=max(a)
    L=len(str(M))
    for i in range(len(a)):
        temp=str(a[i]).zfill(L)
        a[i]=temp
    return a,
def inoutmatrix(a,t):
    matrix=[[]for i in range(10)]    
    for i in range(len(a)-1,-1,-1):
        ind=int(a[i][-t])
        matrix[ind].append(a[i])
    k=0
    for i in range(10):
        if matrix[i]!=[]:
            for j in range(len(matrix[i])-1,-1,-1):
                a[k]=matrix[i][j]
                k+=1
    return a

def radixSort(a):
    M=max(a)
    L=len(str(M))
    for i in range(len(a)):
        temp=str(a[i]).zfill(L)
        a[i]=temp
    for j in range(1,L+1):
        a=inoutmatrix(a,j)
    for i in range(len(a)):
        a[i]=int(a[i])
    return a    