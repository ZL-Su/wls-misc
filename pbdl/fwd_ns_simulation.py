from phi.flow import *
import pylab

DT = 1.5
NU = 0.01

BOX = Box['x,y',0:80,0:100]
INFLOW = CenteredGrid(Sphere(x=30,y=15,radius=10),extrapolation.BOUNDARY, x=32, y=40, bounds=BOX)*2
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=BOX) # sampled at cell centers
velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=BOX) # sampled in staggered form at face centers

def step(velocity:StaggeredGrid, smoke:CenteredGrid, pressure, dt=1.0, buoyancy_factor=1.0):
   smoke=advect.semi_lagrangian(smoke, velocity,dt)+INFLOW
   buoyancy_factor=(smoke*(0, buoyancy_factor)).at(velocity)
   velocity=advect.semi_lagrangian(velocity,velocity,dt)+dt*buoyancy_factor
   velocity=diffuse.explicit(velocity,NU,dt)
   velocity,pressure=fluid.make_incompressible(velocity)
   return velocity,smoke,pressure

for time_step in range(10):
   velocity,smoke,pressure=step(velocity,smoke,None,dt=DT)
   print("Frame {}, max velocity {}".format(time_step,np.asarray(math.max(velocity.values))))
pylab.imshow(np.asarray(smoke.values.numpy('y,x')),origin='lower',cmap='magma')
pylab.show()
