from time import sleep
import clip
import torch
import cv2
import urllib
import os
import numpy as np

from pathlib import Path
from PIL import Image
from nebula3_database.movie_db import MOVIE_DB
from nebula3_database.config import NEBULA_CONF 

class CLIP_API:
    def __init__(self, vlm_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = NEBULA_CONF()
        self.url_prefix = config.get_webserver()
        if vlm_name == 'rn':
            self.model, self.preprocess = clip.load("RN50x64", self.device, download_root="/opt/models/clip_ms") 
        if vlm_name == 'vit':
            self.model, self.preprocess = clip.load("ViT-L/14", self.device, download_root="/opt/models/clip_ms")
        #if vlm_name == 'vid':
        self.nre = MOVIE_DB()
        self.db = self.nre.db
        self.clip_model = clip
        self.temp_file = "/tmp/file.mp4"
    
    def download_video_file(self, movie):
        import cv2
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc'.format(movie)
        cursor = self.db.aql.execute(query)
        url_prefix = self.url_prefix
        url_link = ''
        for doc in cursor:
            url_link = url_prefix+doc['url_path']
            url_link = url_link.replace(".avi", ".mp4")   
            print(url_link)
            urllib.request.urlretrieve(url_link, self.temp_file) 
        video = cv2.VideoCapture(self.temp_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        return(fps, url_link)

    def download_and_get_minfo(self, mid, to_print=False):
        # Download the video locally
        fps, url_link = self.download_video_file(mid)
        movie_info = self.nre.get_movie_info(mid)
        fn = self.temp_file
        if to_print:
            print(f"Movie info: {movie_info}")
            print(f"fn path: {fn}")
        return movie_info, fps, fn

    def _calculate_images_features(self, frame):
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(image)
            image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
            return image_features
        else: 
            return None

    def clip_encode_frame(self, fn, movie_id, scene_element):
        if (fn):
            remote_api = self.nre
            metadata = remote_api.get_movie_info(movie_id)
            mdfs = metadata['mdfs'][scene_element]
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mdfs[2])
                ret, frame_ = cap.read() # Read the frame
                if not ret:
                        print("File not found")
                else:
                    feature_t = self._calculate_images_features(frame_)
                    feature_t /= feature_t.norm(dim=-1, keepdim=True)
            return(feature_t)

    def clip_encode_video(self, movie_id, scene_element):        
        movie_info, fps, fn = self.download_and_get_minfo(movie_id, to_print=True)
        if (fn):
            remote_api = self.nre
            metadata = remote_api.get_movie_info(movie_id)
            mdfs = metadata['mdfs'][scene_element]
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                #Simple MDF search - first, start and middle - to be enchanced 
                print("Scene: ", scene_element )
                cap = cv2.VideoCapture(fn)
                feature_mdfs = []
                for count, mdf in enumerate(mdfs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = cap.read() # Read the frame
                                
                    if not ret:
                        print("File not found")
                    else:
                        feature_t = self._calculate_images_features(frame_)
                        feature_t /= feature_t.norm(dim=-1, keepdim=True)
                        feature_mdfs.append(feature_t.cpu().detach().numpy())   
                feature_mean = np.mean(feature_mdfs, axis=0)
                feature_t = torch.from_numpy(feature_mean)                
                return(feature_t.tolist()[0])                    
        else:
            print("File doesn't exist: ", fn)
    
    def clip_encode_text(self, text):
        text_input = clip.tokenize([str(text)]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        #print(text_features)
        return(text_features.tolist()[0])
    
    def clip_batch_encode_text(self, texts):
        string_texts = []
        for inp in texts:
            string_texts.append(str(inp))
        text_input = clip.tokenize(string_texts).to(self.device)
        batch_features = []
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        for text_feature in text_features:
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            batch_features.append(text_feature.tolist()[0])
            print("One vector", text_feature.tolist()[0])
        print("All vecs ", batch_features)
        return(batch_features)

def main():
    clip=CLIP_API('vit')
    #clip.clip_encode_video('/home/dimas/0028_The_Crying_Game_00_53_53_876-00_53_55_522.mp4','Movies/114207205',0)
    clip.clip_batch_encode_text(["test"])
if __name__ == "__main__":
    main()
