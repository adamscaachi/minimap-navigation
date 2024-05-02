import cv2
import random
import numpy as np

class Generator:

    def __init__(self, field_path):
        self.arrowhead = cv2.resize(cv2.cvtColor(cv2.imread('assets/arrowhead.png', -1), cv2.COLOR_BGR2RGBA), (32,32))
        self.camera = cv2.cvtColor(cv2.imread('assets/camera.png', -1), cv2.COLOR_BGR2RGBA)
        self.monster = cv2.resize(cv2.cvtColor(cv2.imread("assets/monster.png", -1), cv2.COLOR_BGR2RGBA)[24:40, 24:40], (8,8))
        self.field = cv2.cvtColor(cv2.imread(field_path), cv2.COLOR_BGR2RGBA)
        self.backgrounds = self._split_field(10000)

    def _split_field(self, size):
        backgrounds = []
        for i in range(size):
            x = random.randint(0, self.field.shape[0] - 32)
            y = random.randint(0, self.field.shape[1] - 32)
            texture = self.field[x:x+32, y:y+32]
            backgrounds.append(texture)
        return backgrounds

    def _randomise_background(self):
        return self.backgrounds[random.randint(0, 9999)]

    def _add_camera(self, image, angle):
        image = image.copy()
        camera = np.roll(self.camera, -32, axis=0)
        camera_height, camera_width = camera.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((camera_width/2, camera_height/2), -angle, 1)
        rotated_camera = cv2.warpAffine(camera, rotation_matrix, (camera_width, camera_height)) 
        cropped_camera = rotated_camera[16:48,34:66]
        camera_alpha = cropped_camera[:, :, 3] / 255.0 / 2.0
        camera_rgb = cropped_camera[:, :, :3]
        for c in range(3):
            image[:, :, c] = (1 - camera_alpha) * image[:, :, c] + camera_alpha * camera_rgb[:, :, c]
        return image

    def _add_monsters(self, image):
        monster_height, monster_width = self.monster.shape[:2]
        if np.random.randint(5) == 0:
            x = random.randint(0, image.shape[0] - monster_width)
            y = random.randint(0, image.shape[1] - monster_height)
            for i in range(monster_height):
                for j in range(monster_width):
                    if self.monster[i, j, 3] != 0:
                        image[x + i, y + j, 0:3] = self.monster[i, j, 0:3]
        return image

    def _add_arrowhead(self, image):
        angle = random.randint(0, 359)
        arrowhead_height, arrowhead_width = self.arrowhead.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((arrowhead_width // 2, arrowhead_height // 2), angle, 1)
        rotated_arrowhead = cv2.warpAffine(self.arrowhead, rotation_matrix, (arrowhead_width, arrowhead_height))
        y_offset = (image.shape[0] - arrowhead_height) // 2
        x_offset = (image.shape[1] - arrowhead_width) // 2
        for i in range(arrowhead_height):
            for j in range(arrowhead_width):
                if rotated_arrowhead[i, j, 3] != 0:
                    image[y_offset + i, x_offset + j, 0:3] = rotated_arrowhead[i, j, 0:3]
        return image

    def generate_image(self, angle):
        image = self._randomise_background()
        image = self._add_camera(image, angle)
        image = self._add_arrowhead(image)
        image = self._add_monsters(image) 
        return image

    def generate_image_grid(self, rows, cols):
        image = self.generate_image(random.randint(0, 359))
        grid = np.zeros((rows * image.shape[0], cols * image.shape[1], 4), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                x_offset = i * image.shape[1]
                y_offset = j * image.shape[0]
                image = self.generate_image(random.randint(0, 359))
                grid[x_offset:x_offset+image.shape[1], y_offset:y_offset+image.shape[0]] = image
        return grid

    def generate_dataset(self, size):
        images = []
        angles = []
        for i in range(size):
            angle = random.randint(0, 359)
            angles.append(angle)
            image = self.generate_image(angle)
            images.append(image)
        return images, angles