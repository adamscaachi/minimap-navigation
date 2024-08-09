import os
import torch
import shutil
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt

class Evaluator:
    
    def __init__(self, model, state_dict, device, generator):
        self.model = model.to(device)
        self.model.load_state_dict(torch.load(state_dict))
        self.device = device
        self.generator = generator
        if os.path.exists('images'):
            shutil.rmtree('images')
        os.makedirs('images')

    def predict_angle(self, image, recover=True):
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
        x_output, y_output = output.squeeze().tolist()
        if recover == True:
            angle = np.rad2deg(np.arctan2(y_output, x_output)) 
            angle += 360 * (angle < 0)
            return int(angle)
        else:
            return x_output, y_output
        
    def test_all_angles(self):
        targets = np.linspace(0, 359, 18)
        num_samples = 10
        predictions = []
        errors = []
        for target in targets:
            angles = []
            for _ in range(num_samples):
                image = (self.generator.generate_image(target)[:, :, :3].astype(np.float32) / 255.0).transpose(2, 0, 1)
                angles.append(self.predict_angle(image))
            mean_angle = np.mean(angles)
            predictions.append(mean_angle)
            errors.append(np.std(angles) / np.sqrt(num_samples))
        plt.plot((0, 360), (0, 360), 'g-')
        plt.errorbar(targets[1:-1], predictions[1:-1], yerr=errors[1:-1], fmt='o', markersize=5, 
                         markerfacecolor='green', markeredgecolor='black', ecolor='green', markeredgewidth=1)
        plt.xlabel('Vision Cone Angle (°)')
        plt.ylabel(r'$atan2(\beta, \alpha)$')
        plt.savefig('test_all_angles.png', dpi=400, bbox_inches='tight')
        plt.clf()    

    def RMSE(self):
        targets = np.linspace(10, 350, 10000)
        predictions = []
        for target in targets:
            image = (self.generator.generate_image(target)[:, :, :3].astype(np.float32) / 255.0).transpose(2, 0, 1)
            predictions.append(self.predict_angle(image))
        rmse = np.sqrt(np.mean((targets - predictions)**2))
        print(rmse)

    def test_randomly_generated_images(self, num_images):
        for i in range(num_images):
            angle = random.randint(0, 359)
            image = self.generator.generate_image(angle)
            plt.imshow(image)
            plt.axis('off')
            image = (image[:, :, :3].astype(np.float32) / 255.0).transpose(2, 0, 1)
            prediction = self.predict_angle(image)
            plt.text(plt.xlim()[1]*0.975, plt.ylim()[0]*0.975, str(prediction) + '°', ha='right', va='bottom',
                        fontsize=16, color='white', fontweight='bold')
            x_min, x_max = plt.xlim()
            x_padding = 1.5 * (x_max - x_min)
            plt.xlim(x_min - x_padding, x_max + x_padding)
            plt.savefig(f'images/{i}.png', dpi=400, bbox_inches='tight')
            plt.clf()

    def create_gif(self):
        files = [os.path.join('images', f) for f in os.listdir('images')]
        with imageio.get_writer('test_random_images.gif', mode='I', fps=0.5) as writer:
            for file_path in files:
                image = imageio.imread(file_path)
                writer.append_data(image)