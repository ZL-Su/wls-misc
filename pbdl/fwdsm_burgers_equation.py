from phi.flow import *
from phi import __version__
print("Using phiflow version: {}".format(phi.__version__))

N = 128
DX = 2./N
steps = 32
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