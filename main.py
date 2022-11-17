import os
from io import BytesIO
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import face_recognition
import pickle
import dlib
import base64

dlib.DLIB_USE_CUDA=True


def load_base64(base64_string, mode='RGB'):
    """
    이미지 파일을 bytes에서 numpy array로 변환
    """
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    if mode=='RGB':
        image = image.convert('RGB')
    elif mode == 'BGR':
        image = image.convert('RGB')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return np.array(image)


def face_detection(base64_string:bytes, upsample:int):
    '''한장의 이미지에서 얼굴 위치와 얼굴 특징 (128차원 벡터) 추출'''

    image = load_base64(base64_string, mode='RGB')

    face_location = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model='cnn')
    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_location, num_jitters=1, model='small')
    return face_location, face_encoding


def save_face_par(base64_string:bytes, child_ids:str, data_folder:str, class_name:str):
    '''
    부모가 처음 앱에 가입할 때 본인 아이 사진을 업로드 하면 
    그 사진들에서 얼굴을 검출하고 128차원 벡터로 추출하여 파일로 저장
    이미지를 byte-string 형식으로 입력
    사진에서 하나의 얼굴이 검출되어야 함
    '''

    # 얼굴 정보를 저장 할 파일
    data_path = os.path.join(data_folder, class_name) + '.pkl'
    if os.path.isfile(data_path):
        with open(data_path,'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {'known_base64_string':[], 'known_child_ids':[],
                     'known_face_locations':[], 'known_face_encodings':[]}

    # 사진에서 얼굴 위치 및 특징 추출
    if base64_string in data_dict['known_base64_string']:
        return data_dict['known_base64_string'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']
    
    with open(img_path, 'rb') as img:
        base64_string = base64.b64encode(img.read())
    
    face_location, face_encoding = face_detection(base64_string, upsample=1)
    
    if len(face_location) != 1:
        return data_dict['known_base64_string'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']

    data_dict['known_base64_string'].append(base64_string)
    data_dict['known_child_ids'].append(child_ids)
    data_dict['known_face_locations'].append(face_location[0])
    data_dict['known_face_encodings'].append(face_encoding[0])
        
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict['known_base64_string'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']


def remove_face(base64_string:bytes, face_locations:list, child_ids:list, data_folder:str, class_name:str):
    '''저장된 데이터에서 얼굴 정보 삭제'''

    if len(face_locations) != len(child_ids):
        raise Exception('len(face_locations_tag) != len(child_ids_tag)')

    data_path = os.path.join(data_folder, class_name) + '.pkl'
    if not os.path.isfile(data_path):
        raise Exception(f'{data_path} not exist!')

    with open(data_path,'rb') as f:
        data_dict = pickle.load(f)

    for i, face_location in enumerate(face_locations):
        if (base64_string in data_dict['known_base64_string']) and (face_location in data_dict['known_face_locations']) and (child_ids[i] in data_dict['known_child_ids']):
            idx = data_dict['known_face_locations'].index(face_location)
            del data_dict['known_base64_string'][idx]
            del data_dict['known_child_ids'][idx]
            del data_dict['known_face_locations'][idx]
            del data_dict['known_face_encodings'][idx]
    
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)
        

def face_matching(base64_string:bytes, known_face_encodings:list):
    '''단체 사진에서 검출된 얼굴에 대해 이름 태깅'''

    face_locations, face_encodings = face_detection(base64_string, upsample=2)

    face_names = []
    for face_encoding in face_encodings:
        temp = list(face_recognition.face_distance(known_face_encodings, face_encoding))
        dis = [tolerance+1] * len(temp)
        dis[temp.index(min(temp))] = min(temp)
        match =  list(np.array(dis) <= tolerance)

        name = None
        if True in match:
            name = known_child_ids[match.index(True)]

        face_names.append(name)
    return face_locations, face_names


def visualize(base64_string:bytes, output_dir:str, img_name:str, face_locations:list, child_ids:list):
    '''단체 사진에서 얼굴 검출 및 매칭 결과를 이미지로 저장'''

    bgr_img = load_base64(base64_string, mode='BGR')
    for (top, right, bottom, left), name in zip(face_locations, child_ids):
        if not name:
            cv2.rectangle(bgr_img, (left, top), (right, bottom), (255, 0, 0), 2)
            continue

        # 얼굴에 박스 그리기
        cv2.rectangle(bgr_img, (left, top), (right, bottom), (255, 0, 0), 2)

        # 박스 아래 이름 달기
        cv2.rectangle(bgr_img, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(bgr_img, name, (left + 4, bottom + 15), font, 0.4, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, img_name), bgr_img)


if __name__ == '__main__':
    single_img_dir = './img/single'
    group_img_dir = './img/group'
    group_output_dir = './img/output_group'
    
    os.makedirs(group_output_dir, exist_ok=True)

    single_child_ids = os.listdir(single_img_dir)
    group_img_names = os.listdir(group_img_dir)

    tolerance = 0.39

    print('=== face detection and save data ===')
    for child_id in tqdm(single_child_ids):
        single_child_path = os.path.join(single_img_dir, child_id)
        img_names = os.listdir(single_child_path)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(single_child_path, img_name)

            with open(img_path, 'rb') as img:
                base64_string = base64.b64encode(img.read())

            known_base64_string, known_child_ids, known_face_locations, known_face_encodings = save_face_par(base64_string, child_id, './', 'C1')

    print('=== face matching ===')
    for img_name in tqdm(group_img_names):
        img_path = os.path.join(group_img_dir, img_name)

        with open(img_path, 'rb') as img:
            base64_string = base64.b64encode(img.read())
        face_locations, face_names = face_matching(base64_string, known_face_encodings)

        if face_names:
            visualize(base64_string, group_output_dir, img_name, face_locations, face_names)
