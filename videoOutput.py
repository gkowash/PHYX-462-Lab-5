import cv2
import numpy as np
import glob

#GRB color codes
color0 = (180,119,31)  #blue
color1 = (50,107,225)  #red-orange?

name = 'FinalOutput'  # Name of output file
modes = ['color', 'diff', 'delta']  #options: color, diff, delta
showingTracker = [True, True, True]
fps = 60
boxSize = np.array((120,120))  #this has to be copied manually from the analysis.py file


def drawTracker(img, i):
    if i < len(balls[0]):  # Position data only goes to 617, frames for color and diff go to 619
        for ball in balls:
            cv2.circle(img, tuple(ball[i].astype(int)), 6, color0, 2)
            cv2.rectangle(img, tuple((ball[i]-boxSize/2).astype(int)), tuple((ball[i]+boxSize/2).astype(int)), color1, 3)
    else:
        print('Skipped frame ', i)


ball0_pos = np.genfromtxt('data\\ball0_pos.csv', delimiter=',')
#ball0_vel = np.genfromtxt('data\\ball0_vel.csv', delimiter=',')
ball1_pos = np.genfromtxt('data\\ball1_pos.csv', delimiter=',')
#ball1_vel = np.genfromtxt('data\\ball1_vel.csv', delimiter=',')
ball2_pos = np.genfromtxt('data\\ball2_pos.csv', delimiter=',')
#ball2_vel = np.genfromtxt('data\\ball2_vel.csv', delimiter=',')
balls = [ball0_pos, ball1_pos, ball2_pos]

img_array = []
startIndex = [21 + len(mode) for mode in modes]  #starting index for filename slice

for i,mode in enumerate(modes):
    filenames = glob.glob('outputs\\frames\\{}\\*.jpg'.format(mode))
    filenames = sorted(filenames, key=lambda name: int(name[startIndex[i]:-4]))

    print("Filenames ", i, ": ", len(filenames))
    print("Pos size: ", len(balls[0]))

    for j,filename in enumerate(filenames):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        if showingTracker[i]:
            drawTracker(img, j)
        img_array.append(img)
        if j % 10 == 0:
            print("Loaded frame ", i, "-", j)

"""
def drawTracker(img, i):
    for ball in balls:
        cv2.circle(img, tuple(ball[i].astype(int)), 6, color0, 2)
        cv2.rectangle(img, tuple((ball[i]-boxSize/2).astype(int)), tuple((ball[i]+boxSize/2).astype(int)), color1, 3)

for i,filename in enumerate(filenames[:-2]):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    if showingTracker[i]:
        drawTracker(img, i)
    img_array.append(img)
    if i % 10 == 0:
        print("Loaded frame ", i)
"""

print("\nCreating video...")

outName = "{}_{}".format(name, fps)
if len(modes)==1 and showingTracker[0]: outName = outName + "_tracker"
out = cv2.VideoWriter('outputs\\videos\\{}.avi'.format(outName), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
