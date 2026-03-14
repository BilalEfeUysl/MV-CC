import cv2
import os
from tqdm import tqdm

dataset = ['train', 'val', 'test']
for data in dataset:
    directory = f'Data/LEVIR-MCI-dataset/images/{data}/A'
    file_names = os.listdir(directory)

    for file_name in tqdm(file_names):
        if not (file_name.endswith('.png') or file_name.endswith('.jpg')):
            continue
            
        name_without_extension, _ = os.path.splitext(file_name)
        output_path = f'Data/LEVIR-MCI-dataset/images/{data}/video_data/{name_without_extension}.mp4'
        
        # --- İŞTE HAYAT KURTARAN O AKILLI KONTROL BURADA ---
        # Eğer bu mp4 dosyası zaten Drive'ımızda varsa, hiç uğraşma, atla (continue)!
        if os.path.exists(output_path):
            continue 
        # ----------------------------------------------------
        
        image1 = cv2.imread(f'Data/LEVIR-MCI-dataset/images/{data}/A/{file_name}')
        image2 = cv2.imread(f'Data/LEVIR-MCI-dataset/images/{data}/B/{file_name}')
        
        if image1.shape[:2] != image2.shape[:2]:
            raise ValueError("Images must have the same size.")
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, 2.0, (image1.shape[1], image1.shape[0]))
        
        for i in range(8):
            weight = i / 7.0  
            interpolated_frame = cv2.addWeighted(image2, weight, image1, 1 - weight, 0)
            out.write(interpolated_frame)
        out.release()
