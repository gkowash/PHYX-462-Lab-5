import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def bcAvg(data, n):
    output = np.zeros(data.size-n)
    for i in range(data.size-n):
        output[i] = np.mean(data[i:i+n])
    return output

# Plot options
showImg = False  # Show sample frame beneath center-of-mass plot for scale (not complete, probably not going to finish)

# Load data
data = np.genfromtxt('data\\final_data.csv', delimiter=',')
time = np.arange(0,618)*(1/240)  # time in seconds (240 fps)

sns.set()  # Pretty plots

pos = data[:,:6]
vel = data[:,8:] #* -1  # Do I need to invert to make positive=up and negative=down?

px_0 = pos[:,0]
vx_0 = vel[:,0]
py_0 = pos[:,1]
vy_0 = vel[:,1]
px_1 = pos[:,2]
vx_1 = vel[:,2]
py_1 = pos[:,3]
vy_1 = vel[:,3]
px_2 = pos[:,4]
vx_2 = vel[:,4]
py_2 = pos[:,5]
vy_2 = vel[:,5]


n = 1  # Size of boxcar average


# Plot y-positions for all 3 balls
t = time[:py_0.size-n]
plt.plot(t, bcAvg(py_0, n))
plt.plot(t, bcAvg(py_1, n))
plt.plot(t, bcAvg(py_2, n))
plt.suptitle('Height of juggling balls over time')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.show()

# Plot x-positions for all 3 balls
t = time[:px_0.size-n]
plt.plot(bcAvg(px_0, 10))
plt.plot(bcAvg(px_1, 10))
plt.plot(bcAvg(px_2, 10))
plt.suptitle('Horizontal position of juggling balls over time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.show()

# Plot y-velocities for all 3 balls
t = time[:vy_0.size-n]
plt.plot(t, bcAvg(vy_0, n))
plt.plot(t, bcAvg(vy_1, n))
plt.plot(t, bcAvg(vy_2, n))
plt.suptitle('Vertical velocity of juggling balls over time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.show()

# Plot x-velocities for all 3 balls
t = time[:vx_0.size-n]
plt.plot(t, bcAvg(vx_0, n))
plt.plot(t, bcAvg(vx_1, n))
plt.plot(t, bcAvg(vx_2, n))
plt.suptitle('Horizontal velocity of juggling balls over time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.show()



# Plot average position of all 3 balls over time

x_pos = pos[:,[0,2,4]]
y_pos = pos[:,[1,3,5]]
x_avg = np.mean(x_pos, axis=1)
y_avg = np.mean(y_pos, axis=1)
x,y = np.mean(x_avg), np.mean(y_avg)

sns.reset_orig()

if showImg:
    image = mpimg.imread('outputs\\frames\\color\\frame0.jpg')
    image = image + ((255-image)*(4/5)).astype(int)
    x /= 0.0009375  # Conversion factor from m to px
    y = image.shape[0] - y/0.0009375
    x_avg /= 0.0009375
    y_avg = image.shape[0] - y_avg/0.0009375
    plt.imshow(image)
    plt.xlabel('Horizontal position (px)')
    plt.ylabel('Vertical position (px)')
else:
    plt.xlabel('Horizontal position (m)')
    plt.ylabel('Vertical position (m)')

plt.scatter(x, y, c='r', zorder=3)
plt.plot(bcAvg(x_avg,n), bcAvg(y_avg,n))
#plt.xlim(0,0.75)
#plt.ylim(0,0.75)
plt.suptitle('Average position of system')
plt.show()

# Plot deviation from average position over time and print average deviation (requires arrays from previous block)
dist = np.sqrt((x_avg - x)**2 + (y_avg - y)**2)  # Distance of system position from average position
dev = np.mean(dist)  # Average deviation from the average system position
print('Average deviation is ', round(dev,4), 'm')

sns.set()
plt.plot(time[:dist.size-n], bcAvg(dist,n))
plt.suptitle('Deviation from average position of system')
plt.xlabel('Time (s)')
plt.ylabel('Deviation (m)')
plt.show()
