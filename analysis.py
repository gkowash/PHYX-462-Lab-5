import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


class Ball:
    members = []

    def __init__(self, p0, p1, box=(100,100), rad=10):
        self.pos = np.array([p0, p1])   #position in pixels
        self.vel = np.array([self.pos[1]-self.pos[0]])   #velocity in pixels/frame
        self.box = np.array(box)   #box size used to search for pixel locations
        self.rad = rad
        Ball.members.append(self)

        self.pixelCount = 1000  #how many bright pixels are within the search window (initialize to high number)
        self.rect = patches.Rectangle(self.pos[n] - self.box/2, self.box[0], self.box[1], linewidth=1, edgecolor='r', facecolor='none')
        self.circ = patches.Circle(self.pos[n], self.rad)
        ax.add_patch(self.rect)
        ax.add_patch(self.circ)

    def checkOverlap(self, i, threshold=100):
        for ball in Ball.members:
            if ball != self:
                dist = np.sqrt(np.sum( (self.pos[i-1] - ball.pos[i-1])**2 ))
                if dist < 100:
                    return True
        return False

    def checkLock(self, i, threshold=50): #check to see if lock is in danger of losing the ball (originally 400 for "diff" mode)
        if self.pixelCount >= threshold:
            #print("i={}: {}".format(i, self.pixelCount))
            return True
        else:
            print("Not enough pixels in search window for frame ({})".format(i, self.pixelCount))
            return False

    def findBall(self, i, mode='delta'): #diff: difference between R and G, delta: change in diff over time
        if mode == 'delta':
            frame = f_delta[i].astype(float)
        elif mode == 'diff':
            frame = frames[i].f_d.astype(float)
        else:
            print("Invalid mode for Ball.findBall. Use 'delta' or 'diff'.")
        pos_guess = self.pos[i-1] + self.vel[i-2]
        lxb, lyb = (pos_guess - np.array([100,100])/2).astype(int)  #make box size automatic, not hard-corded
        uxb, uyb = (pos_guess + np.array([100,100])/2).astype(int)
        self.pixelCount = np.sum(frame[lyb:uyb,lxb:uxb]/255)

        if self.checkLock(i) and not self.checkOverlap(i):
            new_y,new_x = np.array([lyb,lxb]) + np.mean(np.argwhere(frame[lyb:uyb,lxb:uxb]), axis=0)
            new_pos = np.array([new_x, new_y])
            self.pos = np.append(self.pos, [new_pos], axis=0)
            self.vel = np.append(self.vel, [self.pos[i]-self.pos[i-1]], axis=0)
        else:
            self.pos = np.append(self.pos, [pos_guess], axis=0)   #to avoid overlap issues, maintain constant velocity until out of range
            self.vel = np.append(self.vel, [np.mean(self.vel[i-7:i-2], axis=0)], axis=0)  #average recent velocities to avoid fluctuations

    def draw(self, ax):
        self.rect.set_xy(self.pos[n]-self.box/2)
        self.circ.center = self.pos[n]


class Frame:
    def __init__(self, image):
        self.f = image
        self.f_r = image[:,:,0]
        self.f_g = image[:,:,1]
        self.f_b = image[:,:,2]

        self.f_d = ((self.f_r - self.f_g)) #+ (self.f_r - self.f_b)) / 2. #difference between red vs green and blue values
        self.f_d = self.f_d * (self.f_d > threshold) #remove negative values and those below threshold
        self.f_d = self.f_d * (self.f_d < 220) #a lot of the random noise has very high values, while the signal is much lower
        #self.f_d = (self.f_d > threshold)*255 #set pixels above threshold to white for easier viewing






def bcAvg(data, m, n):
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i,j] = np.mean(data[i:i+m,j:j+n])
    return output

def loadFrames(path, nEnd, nStart=0, cropY=(11,61)):
    frameList = []
    for n in range(nStart, nEnd+1):
        image = mpimg.imread(path+'frame'+str(n)+'.jpg')[cropY[0]:cropY[1],:]
        frameList.append(Frame(image))
    return frameList

def onKeyPress(event):
    print('press', event.key)
    sys.stdout.flush()
    global state
    global n
    if event.key == 'right':
        if n < len(frames)-1:
            n += 1
            print("frame ", n)
    elif event.key == 'left':
        if n > 0:
            n -= 1
            print("frame", n)
    elif event.key == 'c':
        state = c
    elif event.key == 'r':
        state = r
    elif event.key == 'g':
        state = g
    elif event.key == 'b':
        state = b
    elif event.key == 'd':
        state = d
    elif event.key == 't':
        state = t
    update()

def update():
    if state == c:
        image.set_data(frames[n].f)
    elif state == r:
        image.set_data(frames[n].f_r)
    elif state == g:
        image.set_data(frames[n].f_g)
    elif state == b:
        image.set_data(frames[n].f_b)
    elif state == d:
        image.set_data(frames[n].f_d)
    elif state == t: #change over time between frames
        if n < len(frames):
            image.set_data(f_delta[n])  #this may be off by one, but also probably not important
    ball1.draw(ax)
    ball2.draw(ax)
    ball3.draw(ax)
    fig.canvas.draw_idle()


n = 0
r,g,b,c,d,t = 0,1,2,3,4,5
state = c
cropY = (500,1350)
boxSize = (120,120)
mode = 'delta'

if mode == 'delta':
    threshold = 50
elif mode == 'diff':
    threshold = 50 #50
else:
    print("Invalid mode. Choose 'delta' or 'diff'.")

fig, ax = plt.subplots()
frames = loadFrames('frames_Trim2\\', 618, cropY=cropY)   #110 for shorter clip, 618 for longer
f_delta = [abs(frames[i].f_d - np.mean([frames[i-1].f_d, frames[i-2].f_d], axis=0)) for i in range(2,len(frames)-1)]   #change in frame values over time
f_delta.insert(0, abs(frames[1].f_d - frames[0].f_d))  #previous line starts at i=2, so manually add element for i=1

#ball1 = Ball((778,708), (778,708+7), box=boxSize)  #(x,y) coordinates (note that structure of image array is (y,x) )
#ball2 = Ball((438,95), (438-1,95+2), box=boxSize)
#ball3 = Ball((546,547), (546,547-12), box=boxSize)

ball1 = Ball((696,782), (690,780), box=boxSize)  #(x,y) coordinates (note that structure of image array is (y,x) )
ball2 = Ball((566,246), (566,242), box=boxSize)
ball3 = Ball((410,297), (408,306), box=boxSize)

#loop over 3rd frame to the end. updating ball position
if mode == 'diff':
    for i in range(2,len(frames)):
        ball1.findBall(i, mode='diff')
        ball2.findBall(i, mode='diff')
        ball3.findBall(i, mode='diff')
elif mode == 'delta':
    for i in range(2,len(f_delta)):
        ball1.findBall(i, mode='delta')
        ball2.findBall(i, mode='delta')
        ball3.findBall(i, mode='delta')

savingData = False
savingFrames = True
if savingData:
    for i,ball in enumerate(Ball.members):
        np.savetxt('data\\ball{}_pos.csv'.format(i), ball.pos, delimiter=',')
        np.savetxt('data\\ball{}_vel.csv'.format(i), ball.vel, delimiter=',')
if savingFrames:
    for i,frame in enumerate(frames):
        frame_BGR = cv2.cvtColor(frame.f, cv2.COLOR_RGB2BGR)
        cv2.imwrite('outputs\\frames\\color\\frame{}.jpg'.format(i), frame_BGR)
        #cv2.imwrite('outputs\\frames\\diff\\frame{}.jpg'.format(i), frame.f_d)
    #for i,frame in enumerate(f_delta):
    #    cv2.imwrite('outputs\\frames\\delta\\frame{}.jpg'.format(i), frame)

fig.canvas.mpl_connect('key_press_event', onKeyPress)
image = ax.imshow(frames[n].f, cmap='gray')
update()
plt.show()
