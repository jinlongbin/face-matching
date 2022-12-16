import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import face_recognition
import pickle
import dlib


dlib.DLIB_USE_CUDA=True
AIDB_DIR = './'


def load_pickle(class_id: str):
    pickle_path = os.path.join(AIDB_DIR, class_id) + '.pkl'
    if os.path.isfile(pickle_path) == False:
        return {
            "known_img_ids": [],
            "known_child_ids": [],
            "known_face_locations": [],
            "known_face_encodings": [],
        }

    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict


def face_detection(image:np.array, upsample:int, face_location:list=[]):
    '''한장의 이미지에서 얼굴 위치와 얼굴 특징 (128차원 벡터) 추출'''

    resize_rate = 1
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        if image.shape[0] >= image.shape[1]:
            resize_rate = image.shape[0] / 2000
            image = cv2.resize(image, (int(image.shape[1] / resize_rate), 2000))
        else:
            resize_rate = image.shape[1] / 2000
            image = cv2.resize(image, (2000, int(image.shape[0] / resize_rate)))

    if not face_location:
        face_location = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model='cnn')
    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_location, num_jitters=1, model='small')
    return face_location, face_encoding


def save_face_par(image:np.array, class_id:str, child_ids:str, img_id:str):
    '''
    부모가 처음 앱에 가입할 때 본인 아이 사진을 업로드 하면 
    그 사진들에서 얼굴을 검출하고 128차원 벡터로 추출하여 파일로 저장
    이미지를 byte-string 형식으로 입력
    사진에서 하나의 얼굴이 검출되어야 함
    '''

    # 얼굴 정보를 저장 할 파일
    data_path = os.path.join(AIDB_DIR, class_id) + '.pkl'
    data_dict = load_pickle(class_id)

    # 사진에서 얼굴 위치 및 특징 추출
    if img_id in data_dict['known_img_ids']:
        return data_dict['known_img_ids'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']
    
    face_location, face_encoding = face_detection(image, upsample=1)
    
    if len(face_location) != 1:
        return data_dict['known_img_ids'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']

    data_dict['known_img_ids'].append(img_id)
    data_dict['known_child_ids'].append(child_ids)
    data_dict['known_face_locations'].append(face_location[0])
    data_dict['known_face_encodings'].append(face_encoding[0])
        
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict['known_img_ids'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']
        

def save_face_tea(class_id:str, img_id:str, image:np.array, face_locations_tag:list, child_ids_tag:list):
    '''
    선생님이 직접 태깅한 정보를 데이터 베이스에 저장
    한 장의 이미지에서 여러개 얼굴 태깅 가능
    len(face_locations_tag) == len(child_ids_tag)
    '''
    if len(face_locations_tag) != len(child_ids_tag):
        raise Exception('len(face_locations_tag) != len(child_ids_tag)')

    face_locations, face_encodings = face_detection(image, upsample=2, face_location=face_locations_tag)

    # 얼굴 정보를 저장 할 파일
    data_path = os.path.join(AIDB_DIR, class_id) + '.pkl'
    data_dict = load_pickle(class_id)
    
    for i, child_id in enumerate(child_ids_tag):
        if (child_id) and (face_locations_tag[i] not in data_dict['known_face_locations']) and (face_locations_tag[i] in face_locations):
            data_dict['known_img_ids'].append(img_id)
            data_dict['known_child_ids'].append(child_id)
            data_dict['known_face_locations'].append(face_locations_tag[i])
            data_dict['known_face_encodings'].append(face_encodings[face_locations.index(face_locations_tag[i])])
    
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict['known_img_ids'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']


def face_matching(image:np.array, data_dict:dict, tolerance=0.39):
    '''단체 사진에서 검출된 얼굴에 대해 이름 태깅'''

    known_child_ids = data_dict['known_child_ids']
    known_face_encodings = data_dict['known_face_encodings']
    face_locations, face_encodings = face_detection(image, upsample=2)

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


def visualize(image:np.array, output_dir:str, img_name:str, face_locations:list, child_ids:list):
    '''단체 사진에서 얼굴 검출 및 매칭 결과를 이미지로 저장'''

    bgr_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
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
    single_img_dir = '/data/longbin/face_ID/img/221118/single'
    group_img_dir = '/data/longbin/face_ID/img/221118/group'
    group_output_dir = '/data/longbin/face_ID/img/221118/output_group'
    class_id = 'C1'
    
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

            image = face_recognition.load_image_file(img_path)

            known_img_ids, known_child_ids, known_face_locations, known_face_encodings = save_face_par(image, class_id, child_id, img_path[-10:])

    print('=== face matching ===')
    for img_name in tqdm(group_img_names):
        img_path = os.path.join(group_img_dir, img_name)

        image = face_recognition.load_image_file(img_path)
        data_dict = load_pickle(class_id)
        face_locations, face_names = face_matching(image, data_dict, tolerance)

        save_face_tea(class_id, img_path[-10:], image, face_locations, face_names)

        if face_names:
            visualize(image, group_output_dir, img_name, face_locations, face_names)