import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection = '3d')

for i in range(1,11):
   
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    ax.quiver(0,0,0,i,i,i)
    plt.draw()
    plt.pause(.01)
    ax.clear()
# x=[None,None]
# x[1]=1
# print(x[1])
# print(type(x))
x = np.linspace(0, 6*np.pi, 100)
y = np.sin(x)

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

for phase in np.linspace(0, 10*np.pi, 500):
    c=np.sin(x + phase)
    line1.set_ydata(c)
    fig.canvas.draw()
    fig.canvas.flush_events()


x=[1,2]
print(type(x))
y,=x
print(type(y))

x=list(range(0,10))
plt.plot(x)
plt.show()
print(x)
# creating initial data values
# of x and y
x = np.linspace(0, 10, 100)
y = np.sin(x)
P_theta =[90] 
# to run GUI event loop
plt.ion()
#figure, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# ax.set_thetamin(-90)
# ax.set_thetamax(90)
# ax.set_rlim(bottom=-20, top=0)
# ax.set_yticklabels([])
# here we are creating sub plots
# figure, ax = plt.subplots(figsize=(10, 8))
# line1, = ax.plot(x, y)

# figure = plt.figure(figsize=(6,6))
# ax = plt.subplot(111, polar=True)
# line1, = ax.plot([],[])

fig = plt.figure()
ax = fig.add_subplot(polar=True)
x= 90#np.linspace(-2 * np.pi, 2 * np.pi, 200)
new_y=1/2
# setting title
# plt.title("Geeks For Geeks", fontsize=20)
 
# # setting x-axis label and y-axis label
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
 
# Loop
for _ in range(1):
    # creating new Y values
    #new_y = np.sin(x-0.5*_)
 
    # updating data values
   #  line1.set_xdata(x)
   #  line1.set_ydata(new_y)
   #  ax.vlines([90],0,-20)
    plt.show()
    # drawing updated values
    #figure.canvas.draw()
   
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    #figure.canvas.flush_events()
    t=1
    time.sleep(0.1)


# sample_rate = 1e6
# N = 10000 # number of samples to simulate

# # Create a tone to act as the transmitter signal
# t = np.arange(N)/sample_rate # time vector
# f_tone = 0.02e6
# tx = np.exp(2j * np.pi * f_tone * t)

# d = 0.5 # half wavelength spacing
# Nr = 3
# theta_degrees = 20 # direction of arrival (feel free to change this, it's arbitrary)
# theta = theta_degrees / 180 * np.pi # convert to radians
# s = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steering Vector
# print(s) # note that it's 3 elements long, it's complex, and the first element is 1+0j

# s = s.reshape(-1,1) # make s a column vector
# print(s.shape) # 3x1
# tx = tx.reshape(1,-1) # make tx a row vector
# print(tx.shape) # 1x10000

# X = s @ tx # Simulate the received signal X through a matrix multiply
# print(X.shape) # 3x10000.  X is now going to be a 2D array, 1D is time and 1D is the spatial dimension

# n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
# X = X + 0.5*n # X and n are both 3x10000

# w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Conventional, aka delay-and-sum, beamformer
# X_weighted = w.conj().T @ X # example of applying the weights to the received signal (i.e., perform the beamforming)
# print(X_weighted.shape) # 1x10000

# theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
# results = []
# for theta_i in theta_scan:
#    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
#    X_weighted = w.conj().T @ X # apply our weights. remember X is 3x10000
#    results.append(10*np.log10(np.var(X_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
# results -= np.max(results) # normalize (optional)

# # print angle that gave us the max value
# print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

# plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
# plt.xlabel("Theta [Degrees]")
# plt.ylabel("DOA Metric")
# plt.grid()
# plt.show()
# a=(np.array([1,2,3])).reshape(-1,1)
# #a=a.reshape(-1,1)
# print(a)
# plt.plot(a)
# print((a.squeeze()))
# plt.show()
# A = np.random.randn(3,10) # 3x10
# B = np.arange(10) # 1D array of length 10
# print(B.shape)
# B = B.reshape(-1,1) # 10x1
# print(B.shape)
# C = A @ B # matrix multiply
# print(C.shape) # 3x1
# C = C.squeeze() # see next subsection
# print(C.shape) # 1D array of length 3, easier for plotting and other non-matrix Python code


# f_c=10
# Fs = 50*f_c # Hz
# cyc = 2.5
# t = np.arange(0,cyc/f_c,1/Fs) # because our sample rate is 1 Hz
# s = np.sin(f_c*2*np.pi*t)
# N=len(s) # number of points to simulate, and our FFT size
# s = s*np.hamming(N)
# S = np.fft.fftshift(np.fft.fft(s))
# S_mag = np.abs(S)
# S_phase = np.angle(S)
# f = np.arange(Fs/-2, Fs/2, Fs/N)
# plt.figure(0)
# plt.plot(f, S_mag,'.-')
# #plt.figure(1)
# #plt.plot(t,s)
# #plt.plot(f, S_phase,'.-')
# plt.show()