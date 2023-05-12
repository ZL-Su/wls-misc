import os
import visdom
import torch
import numpy as np

# Get available device in this machine
def device():
   if(torch.cuda.is_available()):
      return torch.device("cuda")
   else:
      return torch.device("cpu")
   
def __init_env__(seed:int=42):
   torch.manual_seed(seed)
   str = "Using torch" + torch.__version__
   str = str + " on device " + device().type.capitalize()
   print(str)

def parent_path(path):
   return os.path.abspath(os.path.join(path, os.pardir))

class Visdom():
	def __init__(self, net_type:str, server_port:int=8097):
		self.vis = visdom.Visdom(port=server_port,use_incoming_socket=False)
		self.trainLossInit = True
		self.testLossInit = True
		self.meanVarInit = True
		self.net_type = net_type

	def tile_images(self, images, H, W, HN, WN):
		assert(len(images)==HN*WN)
		images = images.reshape([HN, WN, -1, H, W])
		images = [list(i) for i in images]
		image_blocks = np.concatenate([np.concatenate(row,axis=2) for row in images],axis=1)
		return image_blocks

	def train_loss(self, it, loss):
		loss = float(loss.detach().cpu().numpy())
		if self.trainLossInit:
			self.vis.line(Y=np.array([loss]),X=np.array([it]),win="{0}_trainloss".format(self.net_type),
						  opts={ "title": "{0} (train loss)".format(self.net_type), "markers":"true" })
			self.trainLossInit = False
		else: 
			self.vis.line(Y=np.array([loss]),X=np.array([it]),win=self.net_type+"_trainloss",update="append")

	def test_loss(self, it, loss):
		if self.testLossInit:
			self.vis.line(Y=np.array([loss]), X=np.array([it]),
					  win="{0}_testloss".format(self.net_type),
					  opts={ "title": "{0} (test error)".format(self.net_type) })
			self.testLossInit = False
		else:
			self.vis.line(Y=np.array([loss]),X=np.array([it]),
					  win=self.net_type+"_testloss",update="append")

	def plot_loss(self, type, it, loss):
		if type == "train loss": self.train_loss(it, loss)
		if type == "test loss":  self.test_loss(it, loss)

	def show_image(self, image, rows:int, cols:int, mask=None):
		batch_size = image.size()[0]
		# convert to rgb images if the predicted mask is specified
		if mask != None:
			image_rgb = torch.zeros(batch_size, 3, rows, cols)
			for i in range(0, batch_size):
				rgb = image[i].repeat(3, 1, 1)
				if mask[i] == True : rgb[0] = rgb[2] = 0
				image_rgb[i] = rgb
			image = image_rgb

		if batch_size < 8:
			image = self.tile_images(image, rows, cols, 1, batch_size)
		else:
			image = self.tile_images(image, rows, cols, 8, batch_size//8)
		ntrue = int(mask.sum())
		self.vis.image(image.clip(0, 1), win= "Test Images", 
					  opts={"title": "Test Samples: {0}/{1} - {2}%".format(ntrue, batch_size, ntrue/batch_size*100)})