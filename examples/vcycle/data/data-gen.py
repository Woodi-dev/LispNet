import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sc



def jacobi(v,f,c,size,w=0.6):
    vnew =  v.copy()
    h = 1.0/(size-1)

    for y in range(1,size-1):
        for x in range(1,size-1):
            val =f[y][x]*(h*h)
            val+=v[y-1][x] * c[y-1][x]
            val+=v[y+1][x] * c[y+1][x]
            val+=v[y][x+1] * c[y][x+1]
            val+=v[y][x-1] * c[y][x-1]

            val/=(4.0 * c[y][x])
            vnew[y][x]=val

    return vnew * w + (1-w)*v
def residuum(v,f,c,size):
    r = np.zeros(shape=v.shape)
    h = 1.0/(size-1)

    for y in range(1,size-1):
        for x in range(1,size-1):
            rh =f[y][x]
            val=4*v[y][x] * c[y][x]
            val-=v[y-1][x] * c[y-1][x]
            val-=v[y+1][x] * c[y+1][x]
            val-=v[y][x+1] * c[y][x+1]
            val-=v[y][x-1] * c[y][x-1]

            val/=(h*h)
            rh-=val
            r[y][x]=rh
    return r

def get_radius(x, y, middlex,middley):
    return np.sqrt((x-middlex)**2 + (y-middley)**2)

def addBoundaries(v,config):
    size,size = v.shape
    if config==0:
        for x in range(size):
            v[0][x]=1.0
            v[size-1][x]=1.0
        for y in range(size):
            v[y][0]=1.0
            v[y][size-1]=1.0


def genRightHandside(size ,config):
    f = np.zeros(shape=(size,size),dtype=np.double)
    if config==0:
        pass
    return f

def c_horizontal(x,y,dx,height,offset):
    count=0
    for i in range(1,int(1/dx)):
        if(x<=i*dx +offset):
            break
        else:
            count+=1

    if count%2==0:
        return 1.0
    else:
        return height

def c_vertical(x,y,dy,height,offset):
    count=0
    for i in range(1,int(1/dy)):
        if(y<=i*dy +offset):
            break
        else:
            count+=1

    if count%2==0:
        return 1.0
    else:
        return height


def c_diagonal(x,y,d,height,offset):
    count = 0
    for i in range(1, int(1 / d)):
        if (x+y<=(i*d*2) +offset):
            break
        else:
            count += 1

    if count % 2 == 0:
        return 1.0
    else:
        return height


def genTrainCoefficents(level ,config, flip, h=None,offset=None):

    c = np.zeros(shape=((2**level)+1,(2**level)+1),dtype=np.double)
    height=h
    if h == None:
        height = np.random.randint(8, 16)
    if offset==None:
        offset = np.random.uniform(0,0.2)



    size=(2**level)+1
    h= 1.0/(size-1)

    #horizontal configs
    if config==0:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_horizontal(x*h,y*h,0.5,height,offset)
    if config==1:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_horizontal(x*h,y*h,0.25,height,offset)

    #vertical configs
    if config==2:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_vertical(x*h,y*h,0.5,height,offset)
    if config==3:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_vertical(x*h,y*h,0.25,height,offset)
    #diagonal configs
    if config==4:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_diagonal(x*h,y*h,0.5,height,offset)
    if config==5:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_diagonal(x*h,y*h,0.25,height,offset)

    #diagonal configs
    if config==6:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_diagonal(x*h,y*h,0.5,height,offset)
        c = c[::-1, ::]
    if config==7:
        for y in range(size):
            for x in range(size):
                c[y][x]=c_diagonal(x*h,y*h,0.25,height,offset)
        c = c[::-1, ::]

    if config == 8:
        for y in range(size):
            for x in range(size):
                c[y][x]=1.0
    if flip:
        c=  c[::-1,::-1]
    return c

def genTestCoefficents(level ,config):
    c = np.zeros(shape=((2 ** level) + 1, (2 ** level) + 1), dtype=np.double)

    size = (2 ** level) + 1
    h = 1.0 / (size - 1)

    if config == 0:
        heights = np.random.randint(0, 2, size=(2, 2)) *10+1
        for y in range(size):
            for x in range(size):
                c[y][x] = heights[int((y/size)*2),int((x/size)*2)]

    if config == 1:
        heights = np.random.randint(0, 2, size=(4, 4))*10+1
        for y in range(size):
            for x in range(size):
                c[y][x] = heights[int((y/size)*4),int((x/size)*4)]

    if config == 2:
        heights = np.random.randint(1, 12, size=(2, 2))
        for y in range(size):
            for x in range(size):
                c[y][x] = heights[int((y / size) * 2), int((x / size) * 2)]
    if config == 3:
        heights = np.random.randint(1, 12, size=(4, 4))
        for y in range(size):
            for x in range(size):
                c[y][x] = heights[int((y / size) * 4), int((x / size) * 4)]
    if config == 4:
        circlex = random.uniform(0, 1)
        circley = random.uniform(0, 1)
        radius = 0.5
        for y in range(size):
            for x in range(size):
                hx=h*x
                hy=h*y
                c[y][x] = 1.0
                if(np.sqrt((hx-circlex)**2 + (hy-circley)**2)<radius):
                    c[y][x]=11.0
    if config == 5:
        circle1x = random.uniform(0, 1)
        circle1y = random.uniform(0, 1)

        circle2x = random.uniform(0, 1)
        circle2y = random.uniform(0, 1)
        while(get_radius(circle1x,circle1y,circle2x,circle2y)<=0.6):
            circle2x = random.uniform(0, 1)
            circle2y = random.uniform(0, 1)
        radius = 0.25
        for y in range(size):
            for x in range(size):
                hx=h*x
                hy=h*y
                c[y][x] = 1.0
                if(np.sqrt((hx-circle1x)**2 + (hy-circle1y)**2)<radius):
                    c[y][x]=11.0
                if(np.sqrt((hx-circle2x)**2 + (hy-circle2y)**2)<radius):
                    c[y][x]=11.0

    if config == 6:
        circle1x = random.uniform(0, 1)
        circle1y = random.uniform(0, 1)
        circle2x = random.uniform(0, 1)
        circle2y = random.uniform(0, 1)
        while(get_radius(circle1x,circle1y,circle2x,circle2y)>0.6):
            circle2x = random.uniform(0, 1)
            circle2y = random.uniform(0, 1)
        radius = 0.25


        for y in range(size):
            for x in range(size):
                hx=h*x
                hy=h*y
                c[y][x] = 1.0
                if(np.sqrt((hx-circle1x)**2 + (hy-circle1y)**2)<radius):
                    c[y][x]=11.0
                if(np.sqrt((hx-circle2x)**2 + (hy-circle2y)**2)<radius):
                    c[y][x]=11.0

    return c



def genTrain(batch_size, level):
    size= (2**level)+1
    input=[]

    for b in range(batch_size):
        v = np.random.normal(0.0,0.5,size=(size,size))
        addBoundaries(v,0)
        f = genRightHandside(size,0)
        c = genTrainCoefficents(level,b% 8,int(b/8) %2)
     #   c=genTrainCoefficents(level, 0, False,11,0)
        for i in range(0):
            v = jacobi(v,f,c,33)
        l = [v, f, c]
        result = np.stack(tuple(l),-1)
        input.append(result)
    return np.stack(tuple(input),0)

def genTest(batch_size, level):
    size= (2**level)+1
    input=[]

    for b in range(batch_size):
        v = np.random.normal(0.0,0.5,size=(size,size))
        addBoundaries(v,0)
        f = genRightHandside(size,0)
        c = genTestCoefficents(level,b% 7)
        l=[v,f,c]

        result = np.stack(tuple(l),-1)
        input.append(result)
    return np.stack(tuple(input),0)

def genJumpTest(batch_size, level):
    size= (2**level)+1
    input=[]

    for b in range(batch_size):
        v = np.random.normal(0.0,0.5,size=(size,size))
        addBoundaries(v,0)
        f = genRightHandside(size,0)
        c = genTrainCoefficents(level,0,False,b+1,0)
        l=[v,f,c]

        result = np.stack(tuple(l),-1)
        input.append(result)
    return np.stack(tuple(input),0)

def genJumpExpTest(batch_size, level):
    size = (2 ** level) + 1
    input = []

    for b in range(batch_size):
        v = np.random.normal(0.0, 0.5, size=(size, size))
        addBoundaries(v, 0)
        f = genRightHandside(size, 0)
        c = np.zeros(shape=((2 ** level) + 1, (2 ** level) + 1), dtype=np.double)
        height = 2**b
        offset = 0

        size = (2 ** level) + 1
        h = 1.0 / (size - 1)
        # horizontal configs
        for y in range(size):
            for x in range(size):
                c[y][x] = c_horizontal(x * h, y * h, 1.0 / 2, height, offset)
        l = [v, f, c]

        result = np.stack(tuple(l), -1)
        input.append(result)
    return np.stack(tuple(input), 0)

def genJumpCountTest(batch_size, level):
    size= (2**level)+1
    input=[]

    for b in range(batch_size):
        v = np.random.normal(0.0,0.5,size=(size,size))
        addBoundaries(v,0)
        f = genRightHandside(size,0)
        c = np.zeros(shape=((2 ** level) + 1, (2 ** level) + 1), dtype=np.double)
        height = 10
        offset=0

        size = (2 ** level) + 1
        h = 1.0 / (size - 1)
        # horizontal configs
        for y in range(size):
            for x in range(size):
                c[y][x] = c_horizontal(x * h, y * h, 1.0/(b+1), height, offset)
        l=[v,f,c]

        result = np.stack(tuple(l),-1)
        input.append(result)
    return np.stack(tuple(input),0)
#generate train data
#train = genTrain(450,5)
#generate val data
#val = genTrain(225,5)
#np.save("train3",train)
#np.save("val3",val)


#generate test data
#test = genTest(350,5)
#np.save("test",test)

#jump = genJumpTest(150,5)
#np.save("jump",jump)

#jumpexp = genJumpExpTest(500,5)
#np.save("jumpexp",jumpexp)

#jumpcount= genJumpCountTest(33,5)
#np.save("jumpcount",jumpcount)



level=5
batch_size=20
#c= genTest(batch_size,level)
c=np.load("test.npy")
print(c.shape)
print(c[8])
plt.imshow(c[8,:,:,2])
plt.show()
fig = plt.figure(figsize=(batch_size, level))
cols=5
rows = np.ceil(batch_size/5)



for j in range(batch_size):
    for i in range(0, 1):
        img = (c[j, :, :, 2])
        ax=fig.add_subplot(rows,cols, j + 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        im = plt.imshow(img,cmap='viridis',vmin=1,vmax=11)
        plt.colorbar(im,fraction=0.046, pad=0.04)


plt.show()