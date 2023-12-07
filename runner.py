import os, time, datetime, math, random
from typing import Tuple, List

import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
import timm
import albumentations as A

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XDG_CACHE_HOME"] = "../.cache"


def print_verbose(verbose=True, *args, **kwargs): print(*args) if verbose else None

class LoadDataset(Dataset):
	"""
		Custom dataset on your own, for example got class_1 dir and class_2 dir as input, folder contain images only
	"""
	def __init__(self, 
				 class_1_dir = "./data/Real", 
				 class_2_dir = "./data/Fake"):
		super(LoadDataset, self).__init__()
		self.class_names = ["real", "fake"]
		self.class_num = len(self.class_names)
		self.images = []
		self.labels = []

		for imagename in os.listdir(class_1_dir):
			if self.is_image_file(os.path.join(class_1_dir, imagename)):
				self.images.append(os.path.join(class_1_dir, imagename))
				self.labels.append("real")

		for imagename in os.listdir(class_2_dir):
			if self.is_image_file(os.path.join(class_2_dir, imagename)):
				self.images.append(os.path.join(class_2_dir, imagename))
				self.labels.append("fake")

		self.transform = T.Compose(
			[
				T.Resize((224, 224)),
				T.ToTensor(),
				T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			]
		)
		self.augment = A.Compose(
			[
				A.HorizontalFlip(p = 0.5),
				A.VerticalFlip(p = 0.0),
				A.Rotate(limit = 45, p = 0.5),
				A.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.2, p = 0.0)
			]
		)
  

	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert("RGB")
		label = self.labels[index]

		# Augment
		image_np = np.array(image)
		augmented = self.augment(image=image_np)
		image = Image.fromarray(augmented['image'])

		image_tensor = self.transform(image)

		label = self.class_names.index(label)
		tensor_label = torch.zeros(self.class_num)
		tensor_label[label] = 1.0
	
		return image_tensor, tensor_label

	def __len__(self):
		return len(self.images)

	@staticmethod
	def is_image_file(filename: str):
		return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class BaseModel(nn.Module):
	def __init__(self, pretrained_name, out_channels = 1):
		super(BaseModel, self).__init__()
		self.backbone = timm.create_model(pretrained_name, pretrained = True)
		if pretrained_name in ["mobilenetv3_large_100.miil_in21k_ft_in1k", "mobilenetv3_small_100.lamb_in1k"]:
			self.head = nn.Linear(1000, out_channels)
		else:
			self.head = nn.Linear(1000, out_channels)
	def forward(self, x):
		x = self.backbone(x)
		x = self.head(x)
		return x


class CustomLoss(nn.Module):
	def __init__(self):
		super(CustomLoss, self).__init__()
		self.huber = nn.HuberLoss()
	def forward(self, inputs, targets):
		huber_loss = self.huber(inputs, targets)
		return huber_loss


class ONNXProcessedModel(nn.Module):
	def __init__(self, model):
		super(ONNXProcessedModel, self).__init__()
		self.mean = [0.485, 0.456, 0.406]
		self.std  = [0.229, 0.224, 0.225]
		self.model = model
	def forward(self, x):
		R = (x[:, 0:1, :, :] - self.mean[0]) / self.std[0]
		G = (x[:, 1:2, :, :] - self.mean[1]) / self.std[1]
		B = (x[:, 2:3, :, :] - self.mean[2]) / self.std[2]
		x = torch.cat([R, G, B], dim = 1)
		x = self.model(x)
		x = torch.clamp(x, min=-1.0, max=1.0)
		return x

class Config:
	def __init__(self, epoch = 100, batch_size = 32, num_workers = 8, learning_rate = 1e-4, num_classes = 2):
		self.EPOCH = epoch
		self.BATCH_SIZE = batch_size
		self.NUM_WORKERS = num_workers
		self.LEARNING_RATE = learning_rate
		self.NUM_CLASSES = num_classes


class Runner:
	def __init__(self, 
			  	 train_config: Config = None, 
				 valid_config: Config = None,
				 train_valid_ratio: list = [0.9, 0.1],
				 verbose = True, 
				 pretrained_name = "resnet18",
				 save_path = "./weights/",
				 infer_mode = False):
		
		self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.VERBOSE = verbose

		self.TRAIN_CONFIG = Config(epoch=100, batch_size=16, num_workers=8, learning_rate=1e-4, num_classes=2) if train_config is None else train_config
		print_verbose(self.VERBOSE, f"[INFO] *** TRAIN CONFIG ***")
		print_verbose(self.VERBOSE, f"[INFO] - Epoch: {self.TRAIN_CONFIG.EPOCH}")
		print_verbose(self.VERBOSE, f"[INFO] - Batch size: {self.TRAIN_CONFIG.BATCH_SIZE}")
		print_verbose(self.VERBOSE, f"[INFO] - Num workers: {self.TRAIN_CONFIG.NUM_WORKERS}")
		print_verbose(self.VERBOSE, f"[INFO] - Initial learning rate: {self.TRAIN_CONFIG.LEARNING_RATE}")
		print_verbose(self.VERBOSE, f"[INFO] - Num classes: {self.TRAIN_CONFIG.NUM_CLASSES}")
		
		self.VALID_CONFIG = Config(batch_size=self.TRAIN_CONFIG.BATCH_SIZE, num_workers=self.TRAIN_CONFIG.NUM_WORKERS) if valid_config is None else valid_config
		print_verbose(self.VERBOSE, f"[INFO] *** VALID CONFIG ***")
		print_verbose(self.VERBOSE, f"[INFO] - Batch size: {self.VALID_CONFIG.BATCH_SIZE}")
		print_verbose(self.VERBOSE, f"[INFO] - Num workers: {self.VALID_CONFIG.NUM_WORKERS}")

		self.TRAIN_VALID_RATIO = train_valid_ratio if sum(train_valid_ratio) == 1 else [0.9, 0.1]
		print_verbose(self.VERBOSE, f"[INFO] *** TRAIN VALID RATIO: {self.TRAIN_VALID_RATIO} ***")

		self.train_loader, self.valid_loader = self.create_dataloader() if not infer_mode else (None, None)
		print_verbose(self.VERBOSE, f"[INFO] *** TRAIN LEN: {len(self.train_loader.dataset)} ***") if not infer_mode else None
		print_verbose(self.VERBOSE, f"[INFO] *** VALID LEN: {len(self.valid_loader.dataset)} ***") if not infer_mode else None

		self.MODEL_SAVEPATH = save_path
		os.makedirs(self.MODEL_SAVEPATH, exist_ok=True) if not infer_mode else None

		self.PRETRAINED_NAME = pretrained_name
		self.model = BaseModel(pretrained_name = self.PRETRAINED_NAME, out_channels=self.TRAIN_CONFIG.NUM_CLASSES)
		self.model = self.model.to(self.DEVICE)

		self.loss_fn = CustomLoss() if not infer_mode else None
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.TRAIN_CONFIG.LEARNING_RATE) if not infer_mode else None
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9) if not infer_mode else None
		

	def create_dataloader(self):
		dataset = LoadDataset()
		train_data, valid_data = torch.utils.data.random_split(dataset, self.TRAIN_VALID_RATIO)

		train_loader = DataLoader(train_data, 
								  batch_size=self.TRAIN_CONFIG.BATCH_SIZE, 
								  num_workers=self.TRAIN_CONFIG.NUM_WORKERS, 
								  shuffle=True)
		valid_loader = DataLoader(valid_data, 
							      batch_size=self.VALID_CONFIG.BATCH_SIZE, 
							      num_workers=self.VALID_CONFIG.NUM_WORKERS)

		return train_loader, valid_loader


	def train_step(self, epoch):
		self.model.train()
		train_correct = 0
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data, target = data.to(self.DEVICE), target.to(self.DEVICE)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.loss_fn(output, target)

			loss.backward()
			self.optimizer.step()

			pred = torch.argmax(output, dim=1)
			target = torch.argmax(target, dim=1)
			train_correct += pred.eq(target.view_as(pred)).sum().item()

			if batch_idx % 100 == 0:
				iter_num = batch_idx * len(data)
				total_data = len(self.train_loader.dataset)
				iter_num = str(iter_num).zfill(len(str(total_data)))
				total_percent = 100. * batch_idx / len(self.train_loader)
				print_verbose(self.VERBOSE, f'Train Epoch {epoch + 1}: [{iter_num}/{total_data} ({total_percent:2.0f}%)] | Loss: {loss.item():.10f} | LR: {self.optimizer.param_groups[0]["lr"]:.10f}')

		train_accuracy = 100. * train_correct / len(self.train_loader.dataset)
		return train_accuracy    


	def valid_step(self):
		self.model.eval()
		valid_correct = 0
		for (data, target) in self.valid_loader:
			data, target = data.to(self.DEVICE), target.to(self.DEVICE)
			output = self.model(data)
			print(output)
			print(target)
			pred = torch.argmax(output, dim=1)
			target = torch.argmax(target, dim=1)
			# print(pred)
			# print(target)
			valid_correct += pred.eq(target.view_as(pred)).sum().item()

		valid_accuracy = 100. * valid_correct / len(self.valid_loader.dataset)
		return valid_accuracy


	def train(self):
		max_valid_accuracy = 0.0
		for epoch in range(self.TRAIN_CONFIG.EPOCH):
			tik = time.time()
			train_accuracy = self.train_step(epoch)
			valid_accuracy = self.valid_step()
			print_verbose(self.VERBOSE, f'Training accuracy: {train_accuracy}%')
			print_verbose(self.VERBOSE, f'Validating accuracy: {valid_accuracy}%')
			if valid_accuracy >= max_valid_accuracy:
				max_valid_accuracy = valid_accuracy
				torch.save(self.model.state_dict(), os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_best.pth'))

			if os.path.exists(os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_epoch_{epoch - 10}.pth')):
				os.remove(os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_epoch_{epoch - 10}.pth'))
			torch.save(self.model.state_dict(), os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_epoch_{epoch}.pth'))

			self.scheduler.step()

			tok = time.time()
			runtime = tok - tik
			eta = int(runtime * (self.TRAIN_CONFIG.EPOCH - epoch - 1))
			eta = str(datetime.timedelta(seconds=eta))
			print_verbose(self.VERBOSE, f'Runing time: Epoch {epoch + 1}: {str(datetime.timedelta(seconds=int(runtime)))} | ETA: {eta}')


	def infer(self, weights_path, input_imagepath):
		self.model.load_state_dict(torch.load(weights_path))
		self.model.eval()

		transform = T.Compose(
			[
				T.Resize((224, 224)),
				T.ToTensor(),
				T.Normalize(mean=[0.485, 0.456, 0.406],
							std=[0.229, 0.224, 0.225])
			]
		)

		inputs = Image.open(input_imagepath)
		inputs = transform(inputs)
		inputs = inputs.unsqueeze(0)
		inputs = inputs.to(self.DEVICE)

		outputs = self.model(inputs)
		print(outputs)

	def convert_onnx(self, weights_path):
		self.model.load_state_dict(torch.load(weights_path))
		self.model.eval()

		onnx_model = ONNXProcessedModel(self.model)
		onnx_model.eval()
		onnx_model = onnx_model.to(self.DEVICE)
		dummy_input = torch.randn(4, 3, 224, 224).to(self.DEVICE)
		torch.onnx.export(
			onnx_model,                      # model being run
			dummy_input,                    # model input (or a tuple for multiple inputs)
			f'{weights_path.replace(".pth", ".onnx")}',    # where to save the model (can be a file or file-like object)
			export_params=True,             # store the trained parameter weights inside the model file
			opset_version=11,               # the ONNX version to export the model to
			do_constant_folding=True,       # whether to execute constant folding for optimization
			input_names=['input'],          # the model's input names
			output_names=['output'],        # the model's output names
			dynamic_axes={'input': {0: 'batch_size'},       # variable length axes
						  'output': {0: 'batch_size'}}
		)


if __name__ == "__main__":
	# print([x for x in timm.list_models(pretrained=True) if "mobilenetv3" in x])

	infer_mode = False

	train_config = Config(epoch=100, batch_size=16, num_workers=1, learning_rate=1e-4, num_classes=2)
	runner = Runner(pretrained_name = "mobilenetv3_small_100.lamb_in1k",
				    train_config = train_config,
					save_path = "./weights/",
					infer_mode = infer_mode,
					train_valid_ratio = [0.4, 0.6],)
	runner.train()

