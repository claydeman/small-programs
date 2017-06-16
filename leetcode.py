# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:50:08 2017

@author: 70613
"""
import re
import collections
import functools
from collections import namedtuple
from functools import reduce
from numpy import *
import math
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import tushare as ts
import pandas as pd
#allRange
def AllRange(x,start,length):
    if start==length-1:
        print(x)
    for i in range(start,length):
        if x[start]!=x[i]:
            x[start],x[i]=x[i],x[start]
            AllRange(x,start+1,length)
            x[start],x[i]=x[i],x[start]
#combination?????????????????????????????????????????
def combination(x,start,remaining,length,end,l=[]):     
    if remaining==1:
        for i in range(start+1,end):
            l[length-1]=x[i]
            print(l)        
    else:
        for i in range(start,start+remaining-1):
            l[length-remaining]=x[i]               
            combination(x,i,remaining-i+start-1,length,end,l)
    
#hanoi problem
def hanoi(a,b,c,n):
    if n==1:
        print('move '+a+'->'+c)
    else:
        hanoi(a,c,b,n-1)
        print('move '+a+'->'+c)
        hanoi(b,a,c,n-1)
         
#calcuate 1 numbers 
def calcuate2num1(c):
    num=0
    T=True
    while T==True:        
        if c%2!=0:
            num+=1
        c=int(c/2)
        if c!=0:
            T=True
        else:
            T=False
    return num    
def calcuate2num2(c):
    return bin(c).count("1")

#hanmming weight
def hanmming(a,b):
    c=a^b
    return calcuate2num1(c)
#reverse word sequence
def reverseWords(s):
        """
        :type s: str
        :rtype: str
        """
        l=s.split(" ")
        for i in range(len(l)):
            l[i]=l[i][::-1]
        #for i in range(len(l)-1):
          #  l.insert(2*i+1,' ')
        
        return ' '.join(l)
#reverse string
def reverseStr1(s,k):
    l=len(s)
    d=int(l/(2*k))
    temp=[]
    if len(s)<k:
        s=list(s)
        s.reverse()
        return ''.join(s)
    elif len(s)>=k and len(s)<2*k:    
        t=list(s[:k])
        t.reverse()
        t.extend(list(s[k:]))
        return ''.join(t)
    else:
        for t in range(d):
            st=list(s[t*2*k:t*2*k+k])
            st.reverse()
            st=''.join(st)
            temp.append(st+s[t*2*k+k:(t+1)*2*k])
        lstr=list(s[k*2*d:])
        if len(lstr)<k:
            lstr.reverse()
            lstr=''.join(lstr)
            temp.append(lstr)
        elif len(lstr)>=k and len(lstr)<2*k:           
            tem=lstr[:k]
            tem.reverse()       
            tem.extend(lstr[k:])
            tem=''.join(tem)
            temp.append(tem)
        return ''.join(temp)
    
def reverseStr2(s, k):
    s = list(s)
    for i in range(0, len(s), 2*k):
        s[i:i+k] = reversed(s[i:i+k])
    return "".join(s)            
#reverse number in bit
def findComplement(num):
    return 2**(len(bin(num))-2)-1-num    
        
#find words in same row of keyboard
def findWords1(words):
    newwords=[]
    row1=['q','w','e','r','t','y','u','i','o','p']
    row2=['a','s','d','f','g','h','j','k','l']
    row3=['z','x','c','v','b','n','m']         
    for word in words:   
        key1=0
        key2=0
        key3=0  
        copyword=list(word).copy()         
        word=word.lower()
        word=list(word)
        for j in range(len(word)):
            if word[j] in row1:
                key1+=1
            elif word[j] in row2:
                key2+=1
            elif word[j] in row3 :
                key3+=1
        if key1==len(word) or key2==len(word) or key3==len(word):    
            word=''.join(copyword)
            newwords.append(word)           
    return newwords   
def findWords2(words):
        return filter(re.compile('(?i)([qwertyuiop]*|[asdfghjkl]*|[zxcvbnm]*)$').match, words)    

  #substitude word
def fizzBuzz1(n):
        """
        :type n: int
        :rtype: List[str]
        """
        fbzz=[]
        for i in range(1,n+1):
            if i%3==0:
                if i%5==0:
                    fbzz.append('FizzBuzz')
                else:
                    fbzz.append('Fizz')
            elif i%5==0:
                fbzz.append('Buzz')
            else:                
                fbzz.append(str(i))
        return fbzz
def fizzBuzz2(self, n):
    return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]
#topKFrequent()
def topKFrequent(nums,k):
    result=[0 for i in range(k)]
    Nums=list(set(nums))    
    ind={}
    for i in range(len(Nums)):
        ind[Nums[i]]=nums.count(Nums[i])
    print(ind)
    inde=sorted(ind.items(),key=lambda item:item[1],reverse=True)
    for i in range(k):
        result[i]=inde[i][0]
    print(inde)
    print(result)
    
#find the least bricks
def leastBricks1(wall):
    M=0
    for x in range(len(wall)):
        if M<len(wall[x]):
            M=len(wall[x])
    crossed=[[0 for col in range(M-1)] for row in range(len(wall))]
    for i in range(len(wall)):
        for j in range(len(wall[i])-1):
            crossed[i][j]=sum(wall[i][:j+1])
    crossed=sum(crossed,[])
    ind=set(crossed)
    ind=list(ind)
    MaxFrequence=0
    result=0
    for i in range(len(ind)):
        if ind[i]!=0:
            if crossed.count(ind[i])>MaxFrequence:
                MaxFrequence=crossed.count(ind[i])
                result=MaxFrequence
                
    return len(wall)-result
    
def leastBricks2( wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        d = collections.defaultdict(int)
        for line in wall:
            i = 0
            for brick in line[:-1]:
                i += brick
                d[i] += 1
        # print len(wall), d
        return len(wall)-max(d.values())
#fourSumCount
class Solution(object):
    def fillMap(self,A,B):
        Sum=collections.defaultdict(int)
        for i in range(len(A)):
            for j in range(len(B)):
                Sum[A[i]+B[j]]+=1
        return Sum
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        sum1=self.fillMap(A,B)
        sum2=self.fillMap(C,D)
        res=0
        for x in sum1:
            for y in sum2:
                if x==-y:
                    res+=sum1[x]*sum2[y]
        return res
    '''
#namedtuple using method
websites = [
    ('Sohu', 'http://www.google.com/', u'张朝阳'),
    ('Sina', 'http://www.sina.com.cn/', u'王志东'),
    ('163', 'http://www.163.com/', u'丁磊')
]

#Website = namedtuple('Website', ['name', 'url', 'founder'])
'''

#find the difference

def findTheDifference1(s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        s=list(s)
        t=list(t)
        for x in s:
            if x in t:
                t.remove(x)
        return t
def findTheDifference2(s, t):
        return list((collections.Counter(t) - collections.Counter(s)))[0]
#canWinNim
def canWinNim1(n):
    solve=[]
    for i in range(n):
        for j in range(int((n-i)/2)+1):
            for k in range(int((n-i-2*j)/3)+1):
                if i+2*j+3*k==n:
                    temp=[i,j,k]
                    solve.append(temp)
    Num=0
    for x in solve:                
        if sum(x)!=1 or sum(x)%3!=1:
            Num+=1
    if Num==0:
        return True
    else:
        return False
    

def canWinNim2(n):
    return bool(n%4!=0)


#Given a binary array, find the maximum number of consecutive 1s in this array.
def findMaxConsecutiveOnes(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count=0
        MaxCount=0
        for i in range(len(nums)):
            if nums[i]==1:
                count+=1
                if MaxCount<count:
                    MaxCount=count
            else:
                count=0
        return MaxCount

#Given an array of integers, every element appears twice except for one.
# Find that single one.
def singleNumber(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        numCounts=collections.Counter(nums)
        return [k for k,v in numCounts.items() if v==1][0]
def singleNumber3(nums):
    return 2*sum(set(nums))-sum(nums)
    
def singleNumber4(nums):
    return functools.reduce(lambda x, y: x ^ y, nums)
#Given a word, you need to judge whether the usage of capitals in it is right or not. 
def detectCapitalUse1(word):
     """
     :type word: str
     :rtype: bool
     """
     if word.upper()==word:
         return True
     elif word.lower()==word:
         return True
     elif word[0].upper()+word[1:].lower()==word:
         return True
     else:
         return False
def  detectCapitalUse2(word):
     return word.isupper() or word.islower() or word.istitle()
 
#Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
def findDisappearedNumbers1(nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # For each number i in nums,
        # we mark the number that i points as negative.
        # Then we filter the list, get all the indexes
        # who points to a positive number
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])
            print(nums)

        return [i + 1 for i in range(len(nums)) if nums[i] > 0]
    
def findDisappearedNumbers2(nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        l=[i for i in range(1,len(nums)+1)]
        nums=list(set(nums))
        return  list((collections.Counter(l) - collections.Counter(nums)))[:]

#getsum?????????????????????????????????????????????????????????????????
def getsum(a,b):
    if b==0:
        return a
    temp1=a^b
    temp2=(a&b)<<1
    return getsum(temp1,temp2)
    
#BINARY TREE

class BinaryTree():
    def __init__(self,rootid):
        self.left=None
        self.right=None
        self.rootid=rootid
    
    def getLeftChild(self):
        return self.left
    def getRightChild(self):
        return self.right
    def setNodeValue(self,value):
        self.rootid=value
    def getNodeValue(self):
        return self.rootid
    def insertRightChild(self,newNode):
        if self.right==None:
            self.right=BinaryTree(newNode)
        else:
            tree=BinaryTree(newNode)
            tree.right=self.right
            self.right=tree
    def insertLeftChild(self,newNode):
        if self.left==None:
            self.left=BinaryTree(newNode)
        else:
            tree=BinaryTree(newNode)
            tree.left=self.left
            self.left=tree            

def printTree(tree):
    if tree!=None:
        printTree(tree.getLeftChild())
        print(tree.getNodeValue()+' ')
        printTree(tree.getRightChild())
treelist=[]        
def treetolist(tree):
    if tree!=None:
        treelist.append(treetolist(tree.getLeftChild()))
        treelist.append(tree.getNodeValue())
        treelist.append(treetolist(tree.getRightChild()))
    return treelist
def testTree():
    myTree=BinaryTree('1')
    myTree.insertLeftChild('2')
    myTree.insertRightChild('3')
    myTree.insertRightChild('4')
  #  printTree(myTree)
   
    
#binary tree
def _add(node,v):
    new=[v,[],[]]
    if node:
        left,right=node[1:]
        if not left:
            left.extend(v)          
        elif not right:
            right.extend(v)
        else:
            left.extend(v)
    else:
        node.extend(new)
def binary_tree(s):
    root=[]
    for e in s:
        _add(root,e)
    return root

#commen longest sequences
def lcs(a,b):
    if len(a)==0 or len(b)==0:
        return 0
    elif a[-1]!=b[-1]:
        return max(lcs(a[:-1],b),lcs(a,b[:-1]))
    elif a[-1]==b[-1]:
        return lcs(a[:-1],b[:-1])+1
    
#addDigits

def adddig(num):
    result=0
    while(num>0):
        result+=num%10
        num=int(num/10)
    return result

def addDigits1(num):   
    finalResult=num
    while finalResult>10:
        finalResult=adddig(num)
        num=finalResult
    return finalResult

def addDigits2(num):
    """
    :type num: int
    :rtype: int
    """
    if num == 0 : return 0
    else:return (num - 1) % 9 + 1 
                
#encode into short-url and decode

  
#complex Number
def multiimaginary(a,b):
    temp=a+b
    a=a.replace('i','')
    b=b.replace('i','')
    num=[a,b]
    result=1      
    for x in num:        
            result*=int(x)
    if temp.count('i')==2:
        return str(-result)
    elif temp.count('i')==1:
        return str(result)+'i'
    else:
        return str(result)
def addimaginary(a,b):
    if 'i' not in a and 'i' not in b:
        return str(int(a)+int(b))
    elif 'i' in a:
        if 'i' in b:
            a=a.replace('i','')
            b=b.replace('i','')
            a=[a,b]
            result=reduce( (lambda x, y: int(x) + int(y)), a)
            return str(result)+'i'
        else:
            return b+'+'+a 
    elif 'i' not in a and 'i' in b:
        return a+'+'+b        
def addcomplex(a,b):
    result='0'
    realpart='0'
    imaginarypart='0i'
    if '+' in a:
        if '+' in b:
            a=a.split('+')
            b=b.split('+')
            a.extend(b)
            for x in a:
                if 'i' not in x:
                    realpart=addimaginary(realpart,x)
                else:
                    imaginarypart=addimaginary(imaginarypart,x)
            result=addimaginary(realpart,imaginarypart)
            return result
        else:
            a=a.split('+')
            a.append(b)
            for x in a:
                if 'i' not in x:
                    realpart=addimaginary(realpart,x)
                else:
                    imaginarypart=addimaginary(imaginarypart,x)
            result=addimaginary(realpart,imaginarypart)
            return result
    else:
        if '+' in b:
            b=b.split('+')
            b.append(a)
            for x in b:
                if 'i' not in x:
                    realpart=addimaginary(realpart,x)
                else:
                    imaginarypart=addimaginary(imaginarypart,x)
            result=addimaginary(realpart,imaginarypart)
            return result
        else:
            result=addimaginary(a,b)
            return str(result)
    
            
def multicomplex(a,b):
    a=a.split('+')
    b=b.split('+')
    result='0'
    for x in a:
        for y in b:
            result=addcomplex(result,multiimaginary(x,y))
            print(result)            
    return result



#双指针的使用
def moveZeros(nums):
    i=0
    num=0
    while i<len(nums) and num<len(nums):
        if nums[i]==0:
            for j in range(i+1,len(nums)):
                nums[j-1],nums[j]=nums[j],nums[j-1]   
            num+=1
        else:
            i+=1
            num+=1
            
def moveZeros2(nums):
    cur=0
    last=0
    while cur<len(nums):
        if nums[cur]!=0:
            nums[cur],nums[last]=nums[last],nums[cur]
            last+=1
        cur+=1
#find relative Ranks
def findRelativeRanks(nums):   
    temp=nums.copy()
    temp.sort()
    temp.reverse()
    for i in range(len(temp)):
        ind=nums.index(temp[i])
        if i==0:
            nums[ind]='Gold Medal'
            #continue
        elif i==1:
            nums[ind]='Silver Medal'
            #continue
        elif i==2:
            nums[ind]='Bronze Medal'
            #continue
        else:
            nums[ind]=str(i+1)
           # continue
             


#towSum
def twoSum1(numbers,target):
    cur=1
    last=0
    while cur<len(numbers) and last<len(numbers):
        if numbers[cur]+numbers[last]==target:
            return last+1,cur+1
            break            
        elif numbers[cur]+numbers[last]>target or cur==len(numbers)-1:
            last+=1
            cur=last+1
            continue
        elif cur<len(numbers)-1 and numbers[cur]+numbers[last]<target:
            cur+=1
            continue
        else:
            break
            
def twoSum2( numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """       
        l, r = 0, len(numbers)-1
        while l < r:
            s = numbers[l] + numbers[r]
            if s == target:
                return [l+1, r+1]
            elif s < target:
                l += 1
            else:
                r -= 1
                
def twoSum3(numbers,target):                   
    '''
    binary search
    '''
    for i in range(len(numbers)-1):
        j,r=i+1,len(numbers)-1
        while j<=r:
            mid=j+(r-j)//2
            if numbers[i]+numbers[mid]==target:
                return i+1,mid+1
            elif numbers[i]+numbers[mid]<target:
                j=mid+1
            else:
                r=mid-1
        
             
#findcontentChildren
def findContentChildren(g,s):
    if len(g)==0 or len(s)==0:
        return 0
    g.sort()
    s.sort()  
    count=0 
    i=g[0]    
    for j in s:
        if j<i:
            s.remove(j)
        else:
            break
    if j>=i:
        count+=1
        g.remove(i)
        s.remove(j)
        count+=findContentChildren(g,s)            
    return count
'''
def findContent(g,s):
    count=0
    while(len(g)!=0 or len(s)!=0):
        count+=findContentChildren(g,s)[0]
        g,s=findContentChildren(g,s)[1],findContentChildren(g,s)[2]
        print(count,g,s)
    return count
'''
def findContentChildren2(g,s):
     g.sort()
     s.sort()
     res = 0
     i = 0
     for e in s:
         if i == len(g):
             break
         if e >= g[i]:
             res += 1
             i += 1
     return res
   
            


#intersection

def intersection( nums1, nums2):        
    nums1=list(set(nums1))
    nums2=list(set(nums2))
    result=[]
    for i in range(len(nums1)):
        if nums1[i] in nums2:
            result.append(nums1[i])
    return result

           
#magazine
def canConstruct(ransomNote, magazine):
    ransomNote=[x for x in ransomNote]
    magazine=[x for x in magazine]
    for x in ransomNote:
        if x in magazine:
            magazine.remove(x)
        else:
            return False
    return True                    
                    
#print(canConstruct('aa','aab'))


def firstUniqChar1(s):        
     S=sorted(s)
     temp=list(set(S))
     One=[]
     for x in temp:
         if s.count(x)==1:
             One.append(x)
     if len(One)==0:
        return -1
     ind=len(s)
     for x in One:
         if s.index(x)<ind:
             ind=s.index(x)
     return ind
def firstUniqChar2(s):
    return min([s.find(c) for c in string.ascii_lowercase if s.count(c)==1] or [-1])        
#maxProfit(self, prices):
def maxProfit1(prices):
    result=[]
    i=0
    while i<len(prices)-1:
        temp=[]
        j=i
        if prices[j+1]>prices[j]:
            temp.append(prices[j])
        while(prices[j+1]>=prices[j]):
            temp.append(prices[j+1])
            if j<len(prices)-2:
                j+=1
            else:
                break
        result.append(temp)
        i=j+1
    finalResult=0
    for x in result:
        if len(x)==0:
            continue
        else:
            finalResult+=(x[-1]-x[0])       
    return finalResult
        
#s=[2,1]  
#finalResult=maxProfit1(s)    
def maxProfit2(prices):
    return sum(max(prices[i + 1] - prices[i], 0) for i in range(len(prices) - 1))      
#calculate        
def titleToNumber(s):
    sum=0
    for i in range(len(s)):
        sum+=(26**(len(s)-i-1))*(ord(s[i])-64)
    return sum






