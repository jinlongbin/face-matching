import os
import cv2
import numpy as np
from tqdm import tqdm
import face_recognition
import pickle
import dlib

dlib.DLIB_USE_CUDA=True


def face_detection(img_path:str, upsample:int):
    '''한장의 이미지에서 얼굴 위치와 얼굴 특징 (128차원 벡터) 추출'''

    image = face_recognition.load_image_file(img_path)
    face_location = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model='cnn')
    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_location, num_jitters=1, model='small')
    return face_location, face_encoding


def save_face_par(img_folder:str, child_names:list, data_folder:str, class_name:str):
    '''
    부모가 처음 앱에 가입할 때 본인 아이 사진을 업로드 하면 
    그 사진들에서 얼굴을 검출하고 128차원 벡터로 추출하여 파일로 저장
    업로드 된 사진 들은 img_folder에 저장됨
    사진마다 하나의 얼굴이 검출되어야 함
    child_names list에 하나의 원소가 있으면 img_folder에 사진들이 전부 해당 유아 사진
    child_names list에 원소가 img_folder에 있는 사진개수 만큼 있으면 여러 유아 개별 사진
    '''

    img_names = os.listdir(img_folder)

    # 이미지의 개수와 라벨의 개수가 같은지 체크
    if len(img_names) > 1 and len(child_names) == 1:
        child_names = child_names * len(img_names)
    elif len(img_names) == 0:
        return
    elif len(img_names) != len(child_names):
        raise Exception('Number of images and labels are different.')

    # 얼굴 정보를 저장 할 파일
    data_path = os.path.join(data_folder, class_name) + '.pkl'
    if os.path.isfile(data_path):
        with open(data_path,'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {'known_img_paths':[], 'known_child_names':[],
                     'known_face_locations':[], 'known_face_encodings':[]}

    # 각 사진마다 얼굴 위치 및 특징 추출
    for i, img_name in enumerate(img_names):
        img_path = os.path.join(img_folder, img_name)
        if img_path in data_dict['known_img_paths']:
            continue

        face_location, face_encoding = face_detection(img_path, upsample=1)
        if len(face_location) != 1:
            continue  

        data_dict['known_img_paths'].append(img_path)
        data_dict['known_child_names'].append(child_names[i])
        data_dict['known_face_locations'].append(face_location[0])
        data_dict['known_face_encodings'].append(face_encoding[0])
        
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict['known_img_paths'], data_dict['known_child_names'], data_dict['known_face_locations'], data_dict['known_face_encodings']


def save_face_tea(img_path:str, face_locations_tag:list, child_names_tag:list, data_folder:str, class_name:str):
    '''
    선생님이 직접 태깅한 정보를 데이터 베이스에 저장
    한 장의 이미지에서 여러개 얼굴 태깅 가능
    len(face_locations_tag) == len(child_names_tag)
    '''
    if len(face_locations_tag) != len(child_names_tag):
        raise Exception('len(face_locations_tag) != len(child_names_tag)')

    face_locations, face_encodings = face_detection(img_path, upsample=2)

    # 얼굴 정보를 저장 할 파일
    data_path = os.path.join(data_folder, class_name) + '.pkl'
    if os.path.isfile(data_path):
        with open(data_path,'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {'known_img_paths':[], 'known_child_names':[],
                     'known_face_locations':[], 'known_face_encodings':[]}
    
    for i, child_name in enumerate(child_names_tag):
        if (child_name) and (face_locations_tag[i] not in data_dict['known_face_locations']) and (face_locations_tag[i] in face_locations):
            data_dict['known_img_paths'].append(img_path)
            data_dict['known_child_names'].append(child_name)
            data_dict['known_face_locations'].append(face_locations_tag[i])
            data_dict['known_face_encodings'].append(face_encodings[face_locations.index(face_locations_tag[i])])
    
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict['known_img_paths'], data_dict['known_child_names'], data_dict['known_face_locations'], data_dict['known_face_encodings']


def remove_face(img_path:str, face_locations:list, child_names:list, data_folder:str, class_name:str):
    '''저장된 데이터에서 얼굴 정보 삭제'''

    if len(face_locations) != len(child_names):
        raise Exception('len(face_locations_tag) != len(child_names_tag)')

    data_path = os.path.join(data_folder, class_name) + '.pkl'
    if not os.path.isfile(data_path):
        raise Exception(f'{data_path} not exist!')

    with open(data_path,'rb') as f:
        data_dict = pickle.load(f)

    for i, face_location in enumerate(face_locations):
        if (img_path in data_dict['known_img_paths']) and (face_location in data_dict['known_face_locations']) and (child_names[i] in data_dict['known_child_names']):
            idx = data_dict['known_face_locations'].index(face_location)
            del data_dict['known_img_paths'][idx]
            del data_dict['known_child_names'][idx]
            del data_dict['known_face_locations'][idx]
            del data_dict['known_face_encodings'][idx]
    
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)
        

def face_matching(img_path:str, known_face_encodings:list):
    '''단체 사진에서 검출된 얼굴에 대해 이름 태깅'''

    face_locations, face_encodings = face_detection(img_path, upsample=2)

    face_names = []
    for face_encoding in face_encodings:
        temp = list(face_recognition.face_distance(known_face_encodings, face_encoding))
        dis = [tolerance+1] * len(temp)
        dis[temp.index(min(temp))] = min(temp)
        match =  list(np.array(dis) <= tolerance)

        name = None
        if True in match:
            name = known_child_names[match.index(True)]

        face_names.append(name)
    return face_locations, face_names


def visualize(img_dir:str, img_name:str, output_dir:str, face_locations:list, child_names:list):
    '''단체 사진에서 얼굴 검출 및 매칭 결과를 이미지로 저장'''

    img_path = os.path.join(img_dir, img_name)
    bgr_img = cv2.imread(img_path)
    for (top, right, bottom, left), name in zip(face_locations, child_names):
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

    single_child_names = os.listdir(single_img_dir)
    group_img_names = os.listdir(group_img_dir)

    tolerance = 0.39

    print('=== face detection and save data ===')
    for child_name in tqdm(single_child_names):
        single_child_path = os.path.join(single_img_dir, child_name)

        known_img_paths, known_child_names, known_face_locations, known_face_encodings = save_face_par(single_child_path, [child_name], './', 'C1')

    print('=== face matching ===')
    for img_name in tqdm(group_img_names):
        img_path = os.path.join(group_img_dir, img_name)

        
        face_locations, face_names = face_matching(img_path, known_face_encodings)

        if face_names:
            visualize(group_img_dir, img_name, group_output_dir, face_locations, face_names)