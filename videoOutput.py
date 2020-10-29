import cv2
import numpy as np
import glob

#GRB color codes
color0 = (180,119,31)  #blue
color1 = (50,107,225)  #red-orange?

mode = 'diff'  #options: color, diff, delta
showingTracker = True
fps = 60
boxSize = np.array((120,120))  #this has to be copied manually from the analysis.py file

img_array = []
startIndex = 21 + len(mode)  #starting index for filename slice
filenames = glob.glob('outputs\\frames\\{}\\*.jpg'.format(mode))
filenames = sorted(filenames, key=lambda name: int(name[startIndex:-4]))

ball0_pos = np.genfromtxt('data\\ball0_pos.csv', delimiter=',')
#ball0_vel = np.genfromtxt('data\\ball0_vel.csv', delimiter=',')
ball1_pos = np.genfromtxt('data\\ball1_pos.csv', delimiter=',')
#ball1_vel = np.genfromtxt('data\\ball1_vel.csv', delimiter=',')
ball2_pos = np.genfromtxt('data\\ball2_pos.csv', delimiter=',')
#ball2_vel = np.genfromtxt('data\\ball2_vel.csv', delimiter=',')
balls = [ball0_pos, ball1_pos, ball2_pos]


def drawTracker(img, i):
    for ball in balls:
        cv2.circle(img, tuple(ball[i].astype(int)), 6, color0, 2)
        cv2.rectangle(img, tuple((ball[i]-boxSize/2).astype(int)), tuple((ball[i]+boxSize/2).astype(int)), color1, 3)

for i,filename in enumerate(filenames[:-2]):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    if showingTracker:
        drawTracker(img, i)
    img_array.append(img)
    if i % 10 == 0:
        print("Loaded frame ", i)

print("\nCreating video...")

outName = "{}_{}".format(mode, fps)
if showingTracker: outName = outName + "_tracker"
out = cv2.VideoWriter('outputs\\videos\\{}.avi'.format(outName), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
