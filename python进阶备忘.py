#十六进制0x和八进制0o和二进制0b
a=0xAF
b=0o10
c=0b10
#其他进制转二进制用bin，转八进制用oct，十六进制用hex
#获取ASCII用ord，变符号chr。


#变量名可以用字母，数字和下划线，不能以数字开头


#math包含floor，ceil（处理小数还有内置的round），sqrt等常用的数学库，但它不能处理复数，处理复数需要用cmath。
import math
x1=math.floor(1.36)
x2=math.ceil(1.36)
x3=round(1.36)



#复制列表元素要用分片，不能直接复制，不然指针仍然指向同一个存储空间。
x=[1,2,3]
y=x[:]
#或者使用deepcopy
from copy import deepcopy
y=deepcopy(x)



#sort的key可以放指定函数
x=['aa','a','aaaa']
x.sort(key=len)
print(x)
x=[[6,3],[8,1],[7,2]]
x.sort(key= lambda x:x[1])
print(x)



#一个数的元组加“，”即可。
a=42,
print(a)



#模版字符串
#除了替换说明符%外,可以使用Template进行替换。
print('%s aa %o bb %010.3f'%(1,22,3))
#s字符串，f浮点（宽度，精度，最前的0是补0）
#d十进制，o八进制（带符号u），x十六进制（带符号X），科学计数e/E

from string import Template
s1=Template('$x lalala')#替换字符串
print(s1.substitute(x='P'))

s2=Template('${x}ython')#替换单词的一部分
print(s2.substitute(x='P'))

s3=Template('$$ lalala')#转义$
print(s3.substitute(x='P'))

s4=Template('$ython $lalala')#用字典替换多个
d={'ython':'P','lalala':'PP'}
print(s4.substitute(d))


#get访问字典项更安全，项不存在时不报错返回None
d={}
d.get('a')


#断言检查点
age=1
#age=-1
assert age>0, 'age<0' #不满足时直接报错



#使用del是个好习惯.清除引用也会清除名字本身
del age



#封装。
#继承。注意super返回一个类的超类绑定实例。
#多态。多态实现将不同类型的类的对象进行同样对待的特性，不需要知道对象属于哪个类就能调用



#迭代(Iteration)是从某个地方（比如一个列表）取出一个元素的过程。
#一个实现了__iter__方法的对象是可迭代的，一个实现了__next__方法的对象是迭代器。
class TestIterator:
     value=0
     def __next__(self):
          self.value+=1
          if self.value>10: raise StopIteration
          return self.value
     def __iter__(self):
          return self
test=TestIterator()
list(test)



#生成器(Generators)也是一种迭代器，只能对其迭代一次（并不是一次性生成所有数据存在内存，而是一边运行一边生成值，可以节省大量空间）。
#典型的比较是range和xrange。还有带yield的函数都是生成器。
def test_generator():
     li=[1,2,3]
     for i in li:
          yield i#每次产生一个值，函数就会被冻结
print(list(test_generator()))



#Map会将一个函数映射到一个输入列表的所有元素上。
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
#filter过滤列表中的元素，并且返回一个由所有符合要求的元素所构成的列表
number_list = range(-5, 5)
less_than_zero = filter(lambda x: x < 0, number_list)
print(list(less_than_zero))  
#Reduce对一个列表进行一些计算并返回结果
from functools import reduce
product = reduce( (lambda x, y: x * y), [1, 2, 3, 4] )
#以上三种往往与lambda一起食用（当然复杂情况可以传入func）。



#装饰器(Decorators)修改其他函数的功能/行为的函数，让代码更简短，更Pythonic。
def dec1(arg):
     print('dec1')
def dec2(arg):
     print('dec2')
@dec1 #修饰符号@的作用其实是直接运行了代码中被修饰的函数，而且按照修饰的层级进行了参数传递。
@dec2
def test(arg):
    pass
#效果类似dec1(dec2(test(arg)))
#会输出dec2 dec1，test是最终函数体不可调用等同于pass。
#在授权和日志中应用较多。



#对象自省。dir，type，id，inspect。
print(dir(math))#math库所有的函数，类，变量等列出
#print(help(math))#想要的信息都在help里面....


#slots
class MyClass(object):
     def __init__(self, name, identifier):
          self.name = name
          self.identifier = identifier
          self.set_up()
class MyClass(object):
     #固定属性，使类不再建字典保存而用固定的集合，可以节省内存空间40%-50%（对象很多的时候）
     __slots__ = ['name', 'identifier']
     def __init__(self, name, identifier):
          self.name = name
          self.identifier = identifier
          self.set_up()



#CPython



#最最最最最最常用的库
#sys
import sys
args=sys.argv[1:]
print(args)

#*arg和**kwargs的差别
#它们都是处理不定数量参数的传递，*args只是传参数列表，而**kwargs传键值对字典（适合传入带名字或者其他属性的参数）
#当它们同时使用时的顺序：func(fargs, *args, **kwargs)

#haepq
from heapq import *
from random import shuffle
data=[i for i in range(10)]
shuffle(data)
heap=[]
for n in data:
     heappush(heap,n)
print(heap)
print(heappop(heap),heap)


#deque
from collections import deque
q=deque(range(5))
q.append(5)
q.appendleft(6)
print(q)
q.pop()
q.popleft()
q.rotate(3)
print(q)


#re

#Enumerate，有参数可选的！
for c, value in enumerate(items, 10):#从10开始枚举items
     print(c, value)
