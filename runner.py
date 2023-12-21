import os, time, datetime, math, random
from typing import Tuple, List

import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import timm
import albumentations as A

def print_verbose(verbose=True, *args, **kwargs): print(*args) if verbose else None

def byte2gigabyte(bytes, scale = 1024): return bytes / (scale ** 3)
def byte2megabyte(bytes, scale = 1024): return bytes / (scale ** 2)

class LoadDataset(Dataset):
	"""
		Custom dataset on your own, for example got class_1 dir and class_2 dir as input, folder contain images only
	"""
	def __init__(self, 
				 class_1_dir = "...", 
				 class_2_dir = "...",
				 image_size = 224):
		"""
			Add your image paths and labes to self.images and self.labels, for example:
			'''
				for ids in os.listdir(class_1_dir):
					for filepath in os.listdir(os.path.join(class_1_dir, ids)):
						if self.is_image_file(os.path.join(class_1_dir, ids, filepath)):
							self.images.append(os.path.join(class_1_dir, ids, filepath))
							self.labels.append("class 1")

				for ids in os.listdir(class_2_dir):
					for filepath in os.listdir(os.path.join(class_2_dir, ids)):
						if self.is_image_file(os.path.join(class_2_dir, ids, filepath)):
							self.images.append(os.path.join(class_2_dir, ids, filepath))
							self.labels.append("class 2")
			'''
		"""
		super(LoadDataset, self).__init__()
		self.class_names = ["class 1", "class 2"]
		self.class_num = len(self.class_names)
		self.images = []
		self.labels = []

		for _ in range(10000):
			self.images.append("random_img_path")
			self.labels.append(random.choice(self.class_names))

		self.transform = T.Compose([
			T.Resize((image_size, image_size)),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		self.augment = A.Compose([
			A.HorizontalFlip(p = 0.5),
			A.VerticalFlip(p = 0.0),
			A.Rotate(limit = 45, p = 0.0),
			A.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.2, p = 0.2),
			A.GaussNoise(var_limit=(50.0, 150.0), mean=0, per_channel=True, p=0.5)
		])
  

	def __getitem__(self, index):
		"""
			Get item, for example:
			'''
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
			'''
		"""

		image_tensor = torch.randn((3, 224, 224))
		label = self.class_names.index(self.labels[index])
		tensor_label = torch.zeros(self.class_num)
		tensor_label[label] = 1.0
	
		return image_tensor, tensor_label

	def __len__(self):
		return len(self.images)

	@staticmethod
	def is_image_file(filename: str):
		return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class BaseModel(nn.Module):
	def __init__(self, pretrained_name, out_channels = 1, image_size = 224):
		super(BaseModel, self).__init__()
		
		if pretrained_name in ["vit_small_patch14_reg4_dinov2.lvd142m"]:
			self.backbone = timm.create_model(pretrained_name, pretrained = True, img_size = image_size)
		else:
			self.backbone = timm.create_model(pretrained_name, pretrained = True)

		if pretrained_name in ["vit_small_patch14_reg4_dinov2.lvd142m"]:
			self.head = nn.Linear(384, out_channels)
		elif pretrained_name in ["swin_tiny_patch4_window7_224.ms_in22k"]:
			self.head = nn.Linear(21841, out_channels)
		elif pretrained_name in ["eva02_tiny_patch14_224.mim_in22k"]:
			self.head = nn.Linear(192, out_channels)
		else:
			self.head = nn.Linear(1000, out_channels)
	def forward(self, x):
		x = self.backbone(x)
		x = self.head(x)
		return x


class CustomLoss(nn.Module):
	def __init__(self):
		super(CustomLoss, self).__init__()
		self.ce = nn.CrossEntropyLoss()
	def forward(self, inputs, targets):
		ce_loss = self.ce(inputs, targets)
		return ce_loss


class ONNXProcessedModel(nn.Module):
	def __init__(self, model):
		super(ONNXProcessedModel, self).__init__()
		self.mean = [0.485, 0.456, 0.406]
		self.std  = [0.229, 0.224, 0.225]
		self.model = model
		self.model.eval()
		self.sm = nn.Softmax(dim=-1)
	def forward(self, x):
		x = x / 255.0
		R = (x[:, 0:1, :, :] - self.mean[0]) / self.std[0]
		G = (x[:, 1:2, :, :] - self.mean[1]) / self.std[1]
		B = (x[:, 2:3, :, :] - self.mean[2]) / self.std[2]
		x = torch.cat([R, G, B], dim = 1)
		x = self.model(x)
		x = self.sm(x)
		return x

class Config:
	def __init__(self, epoch = 100, batch_size = 32, num_workers = 8, learning_rate = 1e-4, num_classes = 2, image_size = 224):
		self.EPOCH = epoch
		self.BATCH_SIZE = batch_size
		self.NUM_WORKERS = num_workers
		self.LEARNING_RATE = learning_rate
		self.NUM_CLASSES = num_classes
		self.IMAGE_SIZE = image_size


class Runner:
	def __init__(self, 
			  	 train_config: Config = None, 
				 valid_config: Config = None,
				 train_valid_ratio: list = [0.9, 0.1],
				 verbose: bool = True, 
				 pretrained_name: str = "resnet18",
				 pretrained_path: str = None,
				 save_path: str = "./weights/",
				 save_name: str = None):

		self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.VERBOSE = verbose

		self.TRAIN_CONFIG = Config(epoch=100, batch_size=16, num_workers=8, learning_rate=1e-4, num_classes=2) if train_config is None else train_config
		self.TRAIN_CONFIG.NUM_WORKERS = self.TRAIN_CONFIG.BATCH_SIZE if self.TRAIN_CONFIG.NUM_WORKERS > self.TRAIN_CONFIG.BATCH_SIZE else self.TRAIN_CONFIG.NUM_WORKERS

		self.VALID_CONFIG = Config(batch_size=self.TRAIN_CONFIG.BATCH_SIZE, num_workers=self.TRAIN_CONFIG.NUM_WORKERS) if valid_config is None else valid_config
		self.VALID_CONFIG.NUM_WORKERS = self.VALID_CONFIG.BATCH_SIZE if self.VALID_CONFIG.NUM_WORKERS > self.VALID_CONFIG.BATCH_SIZE else self.VALID_CONFIG.NUM_WORKERS

		self.TRAIN_VALID_RATIO = train_valid_ratio if sum(train_valid_ratio) == 1 else [0.9, 0.1]

		if save_name is not None:
			self.MODEL_SAVEPATH = os.path.join(save_path, save_name)
		else:
			self.MODEL_SAVEPATH = os.path.join(save_path, pretrained_name)
		os.makedirs(self.MODEL_SAVEPATH, exist_ok=True) 

		self.PRETRAINED_NAME = pretrained_name
		self.model = BaseModel(pretrained_name = self.PRETRAINED_NAME, 
							   out_channels=self.TRAIN_CONFIG.NUM_CLASSES,
							   image_size=self.TRAIN_CONFIG.IMAGE_SIZE)
		self.model = self.model.to(self.DEVICE)
		self.model.load_state_dict(torch.load(pretrained_path)) if pretrained_path is not None else None

		
	def create_dataloader(self):
		dataset = LoadDataset(image_size=self.TRAIN_CONFIG.IMAGE_SIZE)
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
		train_total = 0
		train_loss = 0

		total_data = len(self.train_loader.dataset)
		batch_number = total_data // self.TRAIN_CONFIG.BATCH_SIZE 
		num_iter = (epoch - 1) * batch_number

		for batch_idx, (data, target) in enumerate(self.train_loader):
			batch_time_start = time.time()
			data, target = data.to(self.DEVICE), target.to(self.DEVICE)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.loss_fn(output, target)
			train_loss += loss.item()

			loss.backward()
			self.optimizer.step()
			
			pred = torch.argmax(output, dim=1)
			target = torch.argmax(target, dim=1)
			train_correct += pred.eq(target.view_as(pred)).sum().item()
			train_total += target.size(0)

			num_iter += 1
			batch_train_accuracy = 100. * train_correct / train_total
			self.writer.add_scalar(f"Iter/train_accuracy", batch_train_accuracy, num_iter)
			self.writer.add_scalar(f"Iter/train_loss", loss.item(), num_iter)
			self.writer.flush()

			batch_time_end = time.time()

			print_fre = 1 # percent
			if batch_idx % ((batch_number * print_fre // 100) + 1e-8) == 0:
				remain_batch = batch_number - batch_idx
				eta = (batch_time_end - batch_time_start) * (remain_batch + batch_number * (self.TRAIN_CONFIG.EPOCH - epoch))
				eta = str(datetime.timedelta(seconds=eta))
				iter_num = batch_idx * len(data)
				iter_num = str(iter_num).zfill(len(str(total_data)))
				total_percent = 100. * batch_idx / len(self.train_loader)
				
				print_verbose(self.VERBOSE, f'Train Epoch {epoch}: [{iter_num}/{total_data} ({total_percent:2.0f}%)] | Loss: {loss.item():.10f} | Acc: {batch_train_accuracy:.4f} | LR: {self.optimizer.param_groups[0]["lr"]:.10f} | ETA: {eta}')

		train_accuracy = 100. * train_correct / len(self.train_loader.dataset)
		train_loss = train_loss / (batch_idx + 1)
		return train_accuracy, train_loss

	def valid_step(self, epoch):
		self.model.eval()
		valid_correct = 0
		valid_total = 0
		valid_loss = 0

		total_data = len(self.valid_loader.dataset)
		batch_number = total_data // self.VALID_CONFIG.BATCH_SIZE 
		num_iter = (epoch - 1) * batch_number

		for batch_idx, (data, target) in enumerate(self.valid_loader):
			data, target = data.to(self.DEVICE), target.to(self.DEVICE)
			output = self.model(data)
			loss = self.loss_fn(output, target)
			valid_loss += loss.item()

			pred = torch.argmax(output, dim=1)
			target = torch.argmax(target, dim=1)

			valid_correct += pred.eq(target.view_as(pred)).sum().item()
			valid_total += target.size(0)

			num_iter += 1
			batch_valid_accuracy = 100. * valid_correct / valid_total
			self.writer.add_scalar(f"Iter/valid_accuracy", batch_valid_accuracy, num_iter)
			self.writer.add_scalar(f"Iter/valid_loss", loss.item(), num_iter)
			self.writer.flush()

		valid_accuracy = 100. * valid_correct / len(self.valid_loader.dataset)
		valid_loss = valid_loss / (batch_idx + 1)
		return valid_accuracy, valid_loss

	def train_allocate(self):
		self.model.train()
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data, target = data.to(self.DEVICE), target.to(self.DEVICE)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.loss_fn(output, target)
			if batch_idx == 10:
				break

	def valid_allocate(self):
		self.model.eval()
		for batch_idx, (data, target) in enumerate(self.valid_loader):
			data, target = data.to(self.DEVICE), target.to(self.DEVICE)
			output = self.model(data)
			if batch_idx == 10:
				break


	def train(self):
		print_verbose(self.VERBOSE, f"[WARNING] TRAIN CONFIG: num_workers > batch_size ({self.TRAIN_CONFIG.NUM_WORKERS} > {self.TRAIN_CONFIG.BATCH_SIZE}) -> set num_workers to {self.TRAIN_CONFIG.BATCH_SIZE}") if self.TRAIN_CONFIG.NUM_WORKERS > self.TRAIN_CONFIG.BATCH_SIZE else None
		print_verbose(self.VERBOSE, f"[WARNING] VALID CONFIG: num_workers > batch_size ({self.VALID_CONFIG.NUM_WORKERS} > {self.VALID_CONFIG.BATCH_SIZE}) -> set num_workers to {self.VALID_CONFIG.BATCH_SIZE}") if self.VALID_CONFIG.NUM_WORKERS > self.VALID_CONFIG.BATCH_SIZE else None
		print_verbose(self.VERBOSE, f"[INFO] *** TRAIN CONFIG ***")
		print_verbose(self.VERBOSE, f"[INFO] - Epoch: {self.TRAIN_CONFIG.EPOCH}")
		print_verbose(self.VERBOSE, f"[INFO] - Batch size: {self.TRAIN_CONFIG.BATCH_SIZE}")
		print_verbose(self.VERBOSE, f"[INFO] - Num workers: {self.TRAIN_CONFIG.NUM_WORKERS}")
		print_verbose(self.VERBOSE, f"[INFO] - Initial learning rate: {self.TRAIN_CONFIG.LEARNING_RATE}")
		print_verbose(self.VERBOSE, f"[INFO] - Num classes: {self.TRAIN_CONFIG.NUM_CLASSES}")
		print_verbose(self.VERBOSE, f"[INFO] *** VALID CONFIG ***")
		print_verbose(self.VERBOSE, f"[INFO] - Batch size: {self.VALID_CONFIG.BATCH_SIZE}")
		print_verbose(self.VERBOSE, f"[INFO] - Num workers: {self.VALID_CONFIG.NUM_WORKERS}")
		print_verbose(self.VERBOSE, f"[INFO] *** TRAIN VALID RATIO: {self.TRAIN_VALID_RATIO} ***")

		self.train_loader, self.valid_loader = self.create_dataloader()
		print_verbose(self.VERBOSE, f"[INFO] *** TRAIN LEN: {len(self.train_loader.dataset)} ***")
		print_verbose(self.VERBOSE, f"[INFO] *** VALID LEN: {len(self.valid_loader.dataset)} ***")

		self.loss_fn = CustomLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.TRAIN_CONFIG.LEARNING_RATE)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

		self.writer = SummaryWriter()

		max_valid_accuracy = 0.0
		for epoch in range(1, self.TRAIN_CONFIG.EPOCH + 1):
			tik = time.time()
			self.train_allocate()
			self.valid_allocate()
			train_accuracy, train_loss = self.train_step(epoch)
			valid_accuracy, valid_loss = self.valid_step(epoch)
			print_verbose(self.VERBOSE, f'Training accuracy: {train_accuracy}%')
			print_verbose(self.VERBOSE, f'Validating accuracy: {valid_accuracy}%')

			self.writer.add_scalar("@Summary/train_loss", train_loss, epoch)
			self.writer.add_scalar("@Summary/valid_loss", valid_loss, epoch)
			self.writer.add_scalar("@Summary/train_accuracy", train_accuracy, epoch)
			self.writer.add_scalar("@Summary/valid_accuracy", valid_accuracy, epoch)
			self.writer.add_scalar("@Summary/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)
			self.writer.flush()

			if valid_accuracy >= max_valid_accuracy:
				max_valid_accuracy = valid_accuracy
				torch.save(self.model.state_dict(), os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_best.pth'))

			if os.path.exists(os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_epoch_{epoch - 10}.pth')):
				os.remove(os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_epoch_{epoch - 10}.pth'))
			torch.save(self.model.state_dict(), os.path.join(self.MODEL_SAVEPATH, f'{self.PRETRAINED_NAME}_epoch_{epoch}.pth'))

			self.scheduler.step()

			tok = time.time()
			runtime = tok - tik
			eta = int(runtime * (self.TRAIN_CONFIG.EPOCH - epoch))
			eta = str(datetime.timedelta(seconds=eta))
			print_verbose(self.VERBOSE, f'Runing time: Epoch {epoch}: {str(datetime.timedelta(seconds=int(runtime)))} | ETA: {eta}')
		self.writer.close()


	def infer(self, weights_path, input_imagepath):
		self.model.load_state_dict(torch.load(weights_path))
		self.model.eval()

		transform = T.Compose(
			[
				T.Resize((self.TRAIN_CONFIG.IMAGE_SIZE, self.TRAIN_CONFIG.IMAGE_SIZE)),
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
		print(torch.softmax(outputs, -1))


	def average_weights(self, list_weights_path: list, save_path):
		def combine_2_weight(sdA, sdB):
			for key in sdA:
				sdA[key] = sdB[key] + sdA[key]

			return sdA

		sdA = torch.load(list_weights_path[0])
		for weights_path in list_weights_path[1:]:
			sdA = combine_2_weight(sdA, torch.load(weights_path))

		for key in sdA:
			sdA[key] = sdA[key]	/ len(list_weights_path)

		torch.save(sdA, save_path)
		print_verbose(self.VERBOSE, f'Average {len(list_weights_path)} epoch, save at {save_path}')


	def convert_onnx(self, weights_path):
		self.model.load_state_dict(torch.load(weights_path))
		self.model.eval()

		onnx_model = ONNXProcessedModel(self.model)
		onnx_model.eval()
		onnx_model = onnx_model.to(self.DEVICE)
		dummy_input = torch.randn(4, 3, self.TRAIN_CONFIG.IMAGE_SIZE, self.TRAIN_CONFIG.IMAGE_SIZE).to(self.DEVICE)
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

	def infer_onnx(self, onnx_model_path, input_imagepath):
		import onnxruntime
		ort_session = onnxruntime.InferenceSession(onnx_model_path, providers = ['CPUExecutionProvider'])

		img_bgr_cv2 = cv2.imread(input_imagepath, cv2.IMREAD_COLOR)
		img_rgb_cv2 = cv2.cvtColor(img_bgr_cv2, cv2.COLOR_BGR2RGB)
		img_rgb_cv2 = cv2.resize(img_rgb_cv2, (224, 224), interpolation = cv2.INTER_LINEAR)
		img_rgb_cv2 = np.float32(img_rgb_cv2)
		img_rgb_cv2_chw = img_rgb_cv2.transpose(2, 0, 1)
		img_rgb_cv2_bchw = np.expand_dims(img_rgb_cv2_chw, 0)
		ort_inputs = {ort_session.get_inputs()[0].name: img_rgb_cv2_bchw}
		ort_outputs = ort_session.run(None, ort_inputs)
		print(ort_outputs[0][0])


if __name__ == "__main__":
	# print([x for x in timm.list_models(pretrained=True) if "mobile" in x])
	# exit()

	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	os.environ["XDG_CACHE_HOME"] = "../.cache"


	train_config = Config(epoch=100, 
						  batch_size=64, 
						  num_workers=1, 
						  learning_rate=1e-4, 
						  num_classes=2,
						  image_size=224)
	runner = Runner(pretrained_name = "mobilenetv3_small_050.lamb_in1k",
				    train_config = train_config,
					save_path = "./weights/",
					train_valid_ratio = [0.7, 0.3],)
	runner.train()

