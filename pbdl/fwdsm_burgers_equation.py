import pylab
from phi.flow import *
from phi import __version__
print("Using phiflow version: {}".format(phi.__version__))

def plot_state(a, title):
   a = np.expand_dims(a, axis=2)
   for i in range(4):
      a = np.concatenate([a,a], axis=2)
   
   a = np.reshape(a, [a.shape[0], a.shape[1]*a.shape[2]])
   fig, axes = pylab.subplots(1, 1, figsize=(16,5))
   im = axes.imshow(a, origin='upper', cmap='inferno')
   pylab.colorbar(im)
   pylab.xlabel('time')
   pylab.ylabel('x')
   pylab.title(title)
   pylab.show()

N = 512
DX = 2./N
steps = 64
dt = 1./steps
nu = 0.01/(N*np.pi)

init_numpy = np.asarray([-np.sin(np.pi * x) for x in np.linspace(-1+DX/2,1-DX/2,N)])
init = math.tensor(init_numpy, spatial('x'))
print(init)

velocity = CenteredGrid(init, extrapolation.PERIODIC, x=N, bounds=Box['x',-1:1])
vt = advect.semi_lagrangian(velocity, velocity, dt=dt)
print("Velocity tensor shape: " +format(velocity.shape))
print("Velocity tensor type: "  +format(type(velocity.values)))
print("Velocity tensor entries 10 to 14: "  +format(velocity.values.numpy('x')[10:15]))

velocities = [velocity]
age = 0.
for i in range(steps):
   v1 = diffuse.explicit(velocities[-1], nu, dt)
   v2 = advect.semi_lagrangian(v1, v1, dt)
   age += dt
   velocities.append(v2)

print("New velocity content at t={} : {} ".format( age, velocities[-1].values.numpy('x,vector')[0:5] ))

vels = [v.values.numpy('x,vector') for v in velocities]
fig = pylab.figure().gca()
fig.plot(np.linspace(-1,1,len(vels[ 0].flatten())), vels[ 0].flatten(), lw=2, color='blue', label="t=0")
fig.plot(np.linspace(-1,1,len(vels[10].flatten())), vels[10].flatten(), lw=2, color='green', label="t=0.3125")
fig.plot(np.linspace(-1,1,len(vels[20].flatten())), vels[20].flatten(), lw=2, color='cyan', label="t=0.625")
fig.plot(np.linspace(-1,1,len(vels[32].flatten())), vels[32].flatten(), lw=2, color='purple',label="t=1")
pylab.xlabel('x')
pylab.ylabel('u')
pylab.legend()
pylab.show()

vels_img = np.asarray(np.concatenate(vels, axis=-1), dtype=np.float32)
plot_state(vels_img, "Velocity")