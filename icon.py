import pygame
import sys
import time
import random
import numpy as np
#import source.Game
from pygame.locals import *
import os

render_to_gif = False

def get_list_ind_to_len(data, goalsize, maxtry =5000):
    ## assume that the amounts of data are the statistics.
    # keys
    vv = [x for x in data]
    totalPaths = sum([data[x] for x in data])
    ## we will assume goalsize >> keys.
    ## we also will use a weighted average for our estimate of number of paths?
    weightAverage = np.sum(np.array([data[x]*x for x in data])/totalPaths)
    number_of_segements = int(goalsize/weightAverage)
    ## error in the number we will go with is sqrt(N) trakcs? Poisson?
    for i in range(maxtry):
        mytest = np.random.choice(vv, size=(np.random.poisson(number_of_segements)), p=(np.array([data[x] for x in data])/totalPaths))
        if sum(mytest) == goalsize:
            return mytest
    print("WE FAILED")
    return None

def intensity(i, total):
    ## return some value 0. -> 1.
    return float(total-i)/total
    return float(total-1*(4*i-(3*total)))/total

def project_3d_to_2d(v_pnts, o_pos, o_ang, e_vec = np.array([0,0,1])):
    r = (v_pnts.T - o_pos).T
    sx,sy,sz = tuple(np.sin(o_ang))
    cx,cy,cz = tuple(np.cos(o_ang))
    rot= np.array([[cz*cy,sz*cy,-sy],[cz*sy*sx-cx*sz, sz*sy*sx+cx*cz, sx*cy],[cz*cx*sy+sx*sz, sz*sy*cx-sx*cz, cx*cy]])
    d = np.matmul(rot, r)
    d2 = np.vstack([d, np.ones(d.shape[1])])
    f = np.matmul(np.array([[1,0,-e_vec[0]/e_vec[2], 0],
                            [0,1,-e_vec[1]/e_vec[2], 0],
                            [0, 0, 1, 0],
                            [0, 0, 1/e_vec[2], 0]]), d2)
    fw = f[-1,:]+1e-10
    return np.vstack([f[0,:]/f[2,:], f[1,:]/f[2,:]]).T


class HexPath:
    #Goal will be to uniquely enumerate (?) each hexagon path.
    ## then we can call on it in a pattern, or in random order?
    # we would want good enumeration... too. not redudnandt
    def __init__(self, center, size, basecolor = np.array([20,100,230]), max_len = 25, sphere_r = 50, sphere_angle = np.array([0,0]), linewidth = 10, path_override = None):
        self.sphere_r = sphere_r
        self.sphere_angle = sphere_angle
        self.center = center
        self.size = size
        self.max_len = max_len
        self.linewidth = linewidth
        self.basecolor = basecolor
        self.positions = [np.array([0,0])]
        self.deathcount = 0
        self.maxdeath = 10
        self.path_override = path_override
        self.path_step = 0
        self.path_substep = 1
        ## enumerate the grid as... (0,0) -> (    0,       0),  (1,0) -> (    a,0       ), (3*a, 0)
                                  # (0,1) -> (1.5*a, 0.866*a),  (1,1) -> (2.5*a, 0.866*a ), 2,1 -> (4.5*a)
                                  # (0,2) -> (    0, 1.732*a),  (1,0) -> (    a, 1.732*a)

                                  #0,1,3,4,6,7...
                                  #i+floor(i/2)

        self.alive = True
    def step(self):
        if len(self.positions) >= self.max_len:
            self.alive = False
        if not self.alive:
            self.deathcount +=1
            if self.deathcount >= self.maxdeath:
                if self.path_override is None:
                        temppos = self.positions
                        self.__init__(self.center, self.size, self.basecolor, self.max_len, self.sphere_r, self.sphere_angle, self.linewidth)
                        return temppos
                else:
                    self.deathcount = 0
                    self.path_step += 1
                    self.path_substep = 1
                    if self.path_step == len(self.path_override):
                        self.path_step = 0
                    self.positions = [self.path_override[self.path_step][0]]
                    self.alive = True
            ## do death_procedure.
            return False

        if self.path_override is not None:
            if self.path_substep == len(self.path_override[self.path_step]):
                if self.path_substep == self.max_len:
                    self.alive = False
                else:
                    self.alive = False
                    self.deathcount += 1
                return False
            self.positions.append(self.path_override[self.path_step][self.path_substep])
            self.path_substep +=1
            return True

        self.direction = np.arange(3)
        self.myPosition = self.positions[-1]
        posSet = set([tuple(x) for x in self.positions])
        #print(self.myPosition.shape)
        # -1, 0, 1

        # neighbor of ( 0, 0) => ( 1, 0), (-1, 1), (-1,-1) : (-1,-1), ( 1, 0), (-1, 1)
        # neighbor of ( 1, 0) => ( 0, 0), ( 0, 1), ( 0,-1) : (-1,-1), (-1, 0), (-1, 1)
        # neighbor of ( 0, 1) => ( 1, 0), ( 1, 2), ( 1, 1) : ( 1,-1), ( 1, 0), ( 1, 1)
        # neighbor of ( 1, 1) => ( 2, 2), ( 0, 1), ( 2, 0) : ( 1,-1), (-1, 0), ( 1, 1)

        self.test_y = self.direction-1
        self.test_x = np.where(self.direction-1 == 0,  (1-2*(self.myPosition[0]%2)),  (1-2*(1-self.myPosition[1]%2)))

        self.tempPos = self.myPosition+np.vstack((self.test_x, self.test_y)).T
        np.random.shuffle(self.tempPos)

        for point in self.tempPos:
            if tuple(point) in posSet:
                continue
            self.positions.append(point)
            return True

        self.alive = False
        return False
    def get_pos(self, rotation = 0, t= 0):
        self.hexiPathRect = np.array(self.positions, dtype='float64')
        pos = np.array(self.positions)
        self.xpoints = pos[:,0].astype('float64')
        self.ypoints = pos[:,1].astype('float64')
        self.hexiPathRect[:,0][pos[:,1]%2 == 0] = self.size*(np.floor(pos[:,0][pos[:,1]%2 == 0]/2)+pos[:,0][pos[:,1]%2 == 0])
        self.hexiPathRect[:,0][pos[:,1]%2 == 1] = self.size*(1.5+np.floor(pos[:,0][pos[:,1]%2 == 1]/2)+pos[:,0][pos[:,1]%2 == 1])
        self.hexiPathRect[:,1] = 0.86602540378*self.size*pos[:,1]
        #self.hexiPathRect += self.center
        if rotation == 0:
            return self.hexiPathRect+self.center
        #rotation = 0
        xy = np.matmul(np.array([[np.cos(rotation), -np.sin(rotation)],
                                   [np.sin(rotation),  np.cos(rotation)]]), self.hexiPathRect.T).T+self.center
        #xy = self.hexiPathRect
        #sphere_r = 80.*np.cos(t/8)#10.+80.*(np.cos(t/12.))**2
        sphere_r = np.abs(self.sphere_r)
        r, theta = np.sqrt(xy[:,0]**2+xy[:,1]**2)/(4*np.pi*sphere_r), np.arctan2(xy[:,1], xy[:,0])

        my3d = sphere_r*np.vstack([np.sin(r)*np.cos(theta),np.sin(r)*np.sin(theta),np.sign(self.sphere_r)*np.cos(r)])

        cx,cy = tuple(np.cos(self.sphere_angle))
        sx,sy = tuple(np.sin(self.sphere_angle))

        rrmat = np.array([[cy   ,   -sy,  0.],
                          [cx*sy, cx*cy, -sx],
                          [sx*sy, cy*sx,  cx]])

        my3d = np.matmul(rrmat, my3d)
        rotation *= float(1+int(sphere_r/30))

        #test = project_3d_to_2d(np.hstack([xy, np.zeros((xy.shape[0],1)) ]).T, np.array([0,0,-500]), np.array([0,0,0]))
        test = project_3d_to_2d(my3d, np.array([np.sin(rotation)*500,0,500*np.cos(rotation)]), np.array([0,rotation,0]))
        #test = project_3d_to_2d(my3d, np.array([0,0,-100]), np.array([0,0,0]))
        return test*np.array([4000,4000])+np.array([(1920/2,1080/2)])
    def draw(self, surface, rotation = 0, draw_action=True, t=0):
        posToDraw = self.get_pos(rotation = rotation, t = t)

        #cv = self.basecolor*intensity(len(self.positions), self.max_len)
        widths = self.linewidth*intensity(self.deathcount, self.maxdeath)#len(self.positions), self.max_len+5)
        intss = int(100*intensity(self.deathcount, self.maxdeath))
        returnvals = {}
        count = 0
        for xy0,xy1 in zip(posToDraw[:-1], posToDraw[1:]):
            #maxlen = self.max_len
            #color = self.basecolor *(1. if count < maxlen/2 else float(maxlen/2-(count-maxlen/2)/(maxlen/2)))
            cv = pygame.Color(0,0,0)
            #amp = (100-amp)*0.75
            #h1 = int(count*360/(2*self.max_len)+(20*t*360./250))%360
            h1 = int(count*360/(1*self.max_len)+(1*t*360./250))%360

            #x0 = 120*np.cos(-t*2*np.pi*0.01)+(1920/2)
            #y0 = 120*np.sin(-t*2*np.pi*0.01)+(1080/2)
            #r0 = np.array([x0,y0])
            #pos = np.array([(xy0[0]+xy1[0])/2, (xy0[1]+xy1[1])/2])
            #hpos = int(np.sqrt(np.sum((r0-pos)**2))*2)%360
            #hh = int(h1*np.cos(2*np.pi*t/100.)**2+hpos*np.sin(2*np.pi*t/250)**2)%360

            cv.hsva = (h1,65,intss,intss)
            ## let's add the spatial color thing...
            #cv = get_spatial_color((xy0[0]+xy1[0])/2, (xy0[1]+xy1[1])/2, t, intss)
            if draw_action:
                pygame.draw.line(surface, cv, tuple(xy0), tuple(xy1), int(widths))
            else:
                returnvals[(tuple(xy0), tuple(xy1))] = (cv, int(widths))
            count +=1
        return returnvals


def draw_all(surface, myset):
    for key in myset:
        pygame.draw.line(surface, myset[key][0], key[0], key[1], myset[key][1])

def draw_list_hex(surface, list_of_hex, rotation = 0, t= 0):
    #[x.draw(surface, rotation, True, t) for x in list_of_hex]
    #return 0
    ## 1/3rd (0.058)
    listofpoints =  [x.draw(surface, rotation, False, t) for x in list_of_hex]
    st = time.time()
    myset = {}
    for singset in listofpoints:
        for key in singset:
            if key in myset:
                if myset[key][1] < singset[key][1]:
                    myset[key] = singset[key]
                continue
            if (key[1], key[0]) in myset:
                if myset[(key[1], key[0])][1] < singset[key][1]:
                    myset[(key[1], key[0])] = singset[key]
                continue
            myset[key] = singset[key]

    draw_all(surface,myset)

def get_spatial_color(x,y,t,amp):
    x0 = 120*np.cos(-t*2*np.pi*0.01)+(1920/2)
    y0 = 120*np.sin(-t*2*np.pi*0.01)+(1080/2)
    r0 = np.array([x0,y0])
    pos = np.array([x,y])
    #h = (int(360.+(180.*np.cos(np.sum((r0-pos)**2)*2*np.pi/(1000)))))%360
    h = int(np.sqrt(np.sum((r0-pos)**2))*2)%360

    #h = (10*t+int(360.+(180.*np.cos(np.sum((r0-pos)**2)*2.*np.pi/(1000.)))))%360
    #print(h)
    #h = (360*np.exp(-float((x-x0)**2+(y-y0)**2)/(2*100**2)))%360
    temp = pygame.Color(0,0,0)
    #amp = (100-amp)*0.75
    temp.hsva = (h,65,amp,amp)
    return temp

if __name__ == "__main__":
    pygame.mixer.pre_init()#22050, 16, 2, 256)
    pygame.font.init()
    pygame.init()
    font = pygame.font.Font(None, 36)

    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
    screen = pygame.display.set_mode((1920, 1080), pygame.DOUBLEBUF | pygame.DOUBLEBUF | pygame.HWSURFACE |pygame.NOFRAME)#, pygame.FULLSCREEN))
    #source.Game.Game().main(screen)


    fpsClock=pygame.time.Clock()

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    surface.fill((255,255,255))

    pygame.key.set_repeat(1, 40)

    screen.blit(surface, (0,0))
    pygame.display.flip()

    #rangles = np.random.uniform(0,np.pi*2, size=(2,4))
    rangles = np.array([[0,np.pi,0,np.pi],[np.pi,np.pi,0,0]])
    hexagon_paths = []
    stored_vals = []


    huge_path_list = {}
    deathsize = 9
    for jkj in range(20*100):
        pp = HexPath(np.array([0,0]), 1, max_len = 25)
        tt = pp.step()
        count = 0
        while type(tt) is bool:
            tt = pp.step()
            count+=1
            if count > 100:
                print(pp.positions)
                print("what.")
                sys.exit(0)
        ## tt is now not a bool, it has died. we have a position.
        if (len(tt)+deathsize) in huge_path_list:
            huge_path_list[len(tt)+deathsize].append(tt)
        else:
            huge_path_list[len(tt)+deathsize] = [tt]

    #print(len(huge_path_list))
    #for key in huge_path_list:
    #    print(key, len(huge_path_list[key]))
    size_path_list = {}
    for key in huge_path_list:
        size_path_list[key] = len(huge_path_list[key])

    attempts = []
    for i in range(40):
        attempt = get_list_ind_to_len(size_path_list, 1000)
        if attempt is None:
            #print("Couldn't find a good permutation for repetitive")
            continue
        attempts.append(attempt)
    if len(attempts) == 0:
        print("No Permutations AT ALL?!?")
        sys.exit(0)

    numpaths = 40

    mysolutions = []
    for i in range(numpaths):
        thissol = []
        attempt = attempts[np.random.choice(len(attempts))]
        np.random.shuffle(attempt)
        for size in attempt:
            thissol.append(huge_path_list[size][np.random.choice(size_path_list[size])])
        mysolutions.append(thissol)

    #print(sum([len(x) for x in mysolutions[0]]))
    #print(len(mysolutions[0]))
    #print(sum([1 for x in mysolutions[0] if len(x) < 25]))
    ## verify mysolution[0] is valid
    #testpath = HexPath(np.array([0,0]), 1, path_override = mysolutions[0])
    #mylist = []
    #for i in range(5500):
    #    if testpath.path_step == 0 and testpath.path_substep == 1:
    #        print(i)
    #    testpath.step()

    s2 = np.random.choice(len(huge_path_list[25]), size=1)
    special_test = [huge_path_list[25][i] for i in s2]

    for i, special_test in enumerate(mysolutions):
        ss = 30 if (i < int(numpaths/2)) else 60#15-10*int(i/30)
        sv = 60 if (i < int(numpaths/2)) else 80
        hexagon_paths.append(HexPath(np.array([0,0]), sv, max_len = 25, sphere_r = ss, sphere_angle = rangles[:,int(i/(numpaths/4))], linewidth=5, path_override = special_test)) #

        #hexagon_paths.append(HexPath(np.array([40,.866*80.]), 80., sphere_r =50))
        #hexagon_paths.append(HexPath(np.array([0,0]), 80., sphere_r = 10+2*i))
        [x.step() for x in hexagon_paths]
        #stored_vals.append([x.positions[-1] for x in hexagon_paths])
        #hexagon_paths = [x if x.alive else HexPath(np.array([0,0]), 30.) for x in hexagon_paths]

    #print(stored_vals[-1])
    #print(stored_vals[0])
    #hexagon_paths = [ for x in range(100)]

    lasttime = time.clock()
    running = True
    t = 0
    #rangles = np.random.uniform(-0.01,0.01, size=(2,6))
    rangles = np.random.uniform(0,0, size=(2,4))
    while running:

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                running = False
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    running = False
                    sys.exit(0)

        #pressedList = pygame.key.get_pressed()
        #if pressedList[K_UP]or pressedList[K_w]:

        surface.fill((0,0,0))

        #for x in hexagon_paths:
        #    x.draw(surface, t/100.)
        draw_list_hex(surface, hexagon_paths, 2*np.pi*3*t/1000., t)

        [x.step() for x in hexagon_paths]

        #rangles = np.random.uniform(-0.05,0.05, size=(2,4))
        for i,hex_path in enumerate(hexagon_paths):
            if i >= numpaths/2:
                hex_path.size = 70+10*np.cos(2*np.pi*(t*16.)/1000) #np.sign(hex_path.sphere_r)*(60+10*np.cos(t/0.05))
            hex_path.sphere_angle += rangles[:,int(i/(numpaths/4))]

        #hexagon_paths.append(HexPath(np.array([0,0]), 80., sphere_r = ss, sphere_angle = rangles[:,int(i/40)]))

        #hexagon_paths.append(HexPath(np.array([40,.866*80.]), 80., sphere_r =50))
        #hexagon_paths.append(HexPath(np.array([0,0]), 80., sphere_r = 10+2*i))

        sizevals = 70+60*np.sin(2*np.pi*(t*32)/1000)
        #for x in hexagon_paths:
        #    x.size = sizevals
        #hexagon_paths = [x if x.alive else  for x in hexagon_paths]
        screen.blit(surface, (0,0))


        nowtime = time.clock()

        if (nowtime-lasttime) < 1./60:
            time.sleep(1./60-(nowtime-lasttime))
        lasttick = 1./(time.clock()-lasttime)
        lasttime = nowtime

        pygame.display.update()
        if render_to_gif:
            if t >=1:
                if t%10 == 0:
                    print(t)
                pygame.image.save(screen, "render/image_%04d.bmp"%(t))
                if t > 1005:
                    sys.exit(0)
        t+=1
