from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import cv2
import numpy as np
import imutils
from imutils import paths
import os
import face_recognition
import pickle
import configparser


# Функция обработки conf файла
def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    conf_dict = {}
    for section in config.sections():
        for key in config[section]:
            conf_dict[key] = config[section][key]
            print((key, config[section][key]))
    return conf_dict


# Декоратор отслеживания загрузки страницы
def load_check(func):
    def inp_arg(*xpath_arg):
        while True:
            if driver.find_elements('xpath', f'{xpath_arg[0]}'):
                func(xpath_arg)
                break
            else:
                pass

    return inp_arg


@load_check
# Первая часть аутентификации
def auth_step_1(xpath_arg):
    driver.find_element('xpath', '//*[@id="standard_auth_btn"]').click()


@load_check
# Вторая часть аутентификации
def auth_step_2(*xpath_arg):
    driver.find_element('xpath', '//*[@id="username"]').send_keys(xpath_arg[0][1])
    driver.find_element('xpath', '//*[@id="password"]').send_keys(xpath_arg[0][2])
    # driver.find_element('xpath', '//*[@id="username"]').send_keys('+79108635290')
    # driver.find_element('xpath', '//*[@id="password"]').send_keys('89108635290Maxim.')
    driver.find_element('xpath', '/html/body/div[1]/main/section[2]/div/div/div/form/div[3]/label').click()
    driver.find_element('xpath', '//*[@id="kc-login"]').click()


@load_check
# Третья часть аутентификации
def auth_step_3(xpath_arg):
    driver.find_element('xpath',
                        '/html/body/div/main/div/main/div[2]/div[1]/main/div/div/div[1]/div[4]/div[4]/div').click()
    pass


@load_check
# Четвертая часть аутентификации
def auth_step_4(xpath_arg):
    driver.find_element('xpath', '/html/body/div/main/div/main/div[1]/div/a[2]').click()
    pass


def user_auth():
    try:
        print('Переходим на Ростелеком ключ')
        driver.get("https://key.rt.ru/main/devices")
        # time.sleep(4)

        print('Начинаем авторизацию')
        auth_step_1('//*[@id="standard_auth_btn"]')

        cur_dir = os.getcwd()
        config = read_ini(cur_dir + '/conf.ini')

        print('Вводим учётные данные')
        auth_step_2('//*[@id="username"]', config['username'], config['password'])

        print('Подключаемся к камере')
        auth_step_4('/html/body/div/main/div/main/div[1]/div/a[2]')

        print('Загружаем видеопоток')
        auth_step_3('/html/body/div/main/div/main/div[2]/div[1]/main/div/div/div[1]/div[4]/div[4]')
        time.sleep(5)

        print('Погнали')
    except Exception as e:
        print(e)
        driver.close()
        driver.quit()


def stream_catch():
    try:
        stream = driver.find_element('xpath',
                                     '/html/body/div/main/div/main/div[2]/div[1]/main/div/div/div[1]')
    except Exception as e:
        print(e)
        return None

    if stream.size['height'] > 400:
        return stream.screenshot_as_png
    else:
        print('Размер полученного изображения некорректен')


def img_decode(raw_img_arg):
    dec_img = np.frombuffer(raw_img_arg, dtype=np.uint8)
    dec_img = cv2.imdecode(dec_img, flags=cv2.COLOR_RGB2GRAY)
    return dec_img


def img_size(orig_img_arg, des_width, mul_acc=False, grey=True):
    res_img = imutils.resize(orig_img_arg, width=des_width)

    if mul_acc:
        x_mul = orig_img_arg.shape[0] / res_img.shape[0]
        y_mul = orig_img_arg.shape[1] / res_img.shape[1]
        return x_mul, y_mul
    else:
        if grey:
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
        return res_img


def dir_check(obj_name='Strangers'):
    cur_directory = os.getcwd()
    if not os.path.exists(f'{cur_directory}/face_dataset/'):
        os.makedirs(f'{cur_directory}/face_dataset/')

    if not os.path.exists(f'{cur_directory}/face_dataset/{obj_name}/'):
        os.makedirs(f'{cur_directory}/face_dataset/{obj_name}/')

    if not os.listdir(f'{cur_directory}/face_dataset/{obj_name}//'):
        return 0
    else:
        dataset_list = os.listdir(f'{cur_directory}/face_dataset/{obj_name}//')
        count_list = []
        for photo in dataset_list:
            count_list.append(photo.replace(".jpg", "").replace("face_", ""))
        return len(count_list)


def face_train():
    # в директории Images хранятся папки со всеми изображениями

    cur_directory = os.getcwd()
    dirs_of_photos = os.listdir(f'{cur_directory}/face_dataset/')
    if dirs_of_photos[0].find('.') != -1:
        dirs_of_photos.pop(0)

    # paths=[]
    # for name in dirs_of_photos:
    #     paths.append(f'{cur_directory}/face_dataset/{name}/')

    imagePaths = list(paths.list_images('face_dataset'))
    knownEncodings = []
    knownNames = []
    # перебираем все папки с изображениями
    for (i, imagePath) in enumerate(imagePaths):
        # извлекаем имя человека из названия папки
        name = imagePath.split(os.path.sep)[-2]
        # загружаем изображение и конвертируем его из BGR (OpenCV ordering)
        # в dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # используем библиотеку Face_recognition для обнаружения лиц
        boxes = face_recognition.face_locations(rgb, model='hog')
        # вычисляем эмбеддинги для каждого лица
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    # сохраним эмбеддинги вместе с их именами в формате словаря
    data = {"encodings": knownEncodings, "names": knownNames}
    # для сохранения данных в файл используем метод pickle
    f = open("face_enc", "wb")
    f.write(pickle.dumps(data))
    f.close()
    print('Набор лиц обучен')


def face_location(frame_arg, x_scale, y_scale):
    faces = face_cascade.detectMultiScale(frame_arg)
    face_coords = [[], [], [], []]
    for x, y, width, height in faces:
        # cv2.rectangle(frame_arg, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)
        faceROI = frame_arg[y:y + height, x:x + width]
        eyes = eye_cascade.detectMultiScale(faceROI)
        for x1, y1, width1, height1 in eyes:
            if x1 >= int(0.08 * width) * 0 and y1 <= int(0.65 * height) and (x1 + width1) <= int(width) \
                    and (y1 + height1) >= int(height * 0.2):
                box = [[x * x_scale], [y * y_scale], [width * x_scale], [height * y_scale]]
                face_coords = np.hstack((face_coords, box))

    return face_coords


def face_recogn_fun(frame_arg, box_arg, x_mul, y_mul):
    rgb1 = cv2.cvtColor(frame_arg, cv2.COLOR_BGR2RGB)
    names = []
    name = ''
    for i in range(len(box_arg[0])):
        box = [0, 0, 0, 0]
        box[0] = box_arg[0][i] / x_mul
        box[1] = box_arg[1][i] / y_mul
        box[2] = box_arg[2][i] / x_mul
        box[3] = box_arg[3][i] / y_mul
        faceROI = rgb1[int(box[1]): int(box[1] + box[3]),
                  int(box[0]): int(box[0] + box[2])]

        encodings = face_recognition.face_encodings(faceROI)

        if data == None:
            return ["Unknown"]

        # name = ""
        for encoding in encodings:
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely
            # and False for rest
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            # set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # Find positions at which we get True and store them
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for j in matchedIdxs:
                    # Check the names at respective indexes we stored in matchedIdxs
                    name = data["names"][j]
                    # increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                # set name which has highest count
                name = max(counts, key=counts.get)
        names.append(name)
    return names


def name_prints(frame_arg, box, names):
    if names:
        for i in range(len(box[0])):
            print(names[i])
            cv2.putText(frame_arg, f"{names[i]}", (int(box[0][i] - 10), int(box[1][i] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# Функция рисования квадратов лиц
def face_box(frame_arg, box):
    # print(box)
    for i in range(len(box[0])):
        cv2.rectangle(frame_arg, (int(box[0][i]), int(box[1][i])),
                      (int(box[0][i] + box[2][i]), int(box[1][i] + box[3][i])),
                      color=(255, 0, 0), thickness=2)
    # функционал для рисования области глаз и имён

    # cv2.rectangle(orig_frame_arg, (int(x + 0.08 * width * 0), int(y + 0.65 * height)),
    #               (int(x + width * 0.9 / 0.9), int(y + height * 0.2)), color=(255, 0, 0), thickness=2)
    # cv2.rectangle(orig_frame_arg, (int(x + x1), int(y + y1)), (int(x + x1 + width1), int(y + y1 + height1)),
    #               color=(0, 255, 0), thickness=2)
    # cv2.putText(orig_frame_arg, f"GAY {name}".format(face_count), (int(x - 10), int(y - 10)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def save_face_box(frame_arg, box_arg, num_face, person='Strangers', x_mul=1, y_mul=1):
    # save_img = cv2.cvtColor(frame_arg, cv2.COLOR_BGR2GRAY)
    cur_directory = os.getcwd()
    box = [0, 0, 0, 0]
    for i in range(len(box_arg[0])):
        box[0] = box_arg[0][i] / x_mul
        box[1] = box_arg[1][i] / y_mul
        box[2] = box_arg[2][i] / x_mul
        box[3] = box_arg[3][i] / y_mul

        faceROI = frame_arg[int(box[1]): int(box[1] + box[3]),
                  int(box[0]): int(box[0] + box[2])]

        cv2.imwrite(f'{cur_directory}/face_dataset/{person}/face_{num_face + i}.jpg', faceROI)
    return num_face + len(box_arg[0])


def home_cam(save_face, desired_width=700):
    user_auth()

    raw_img = stream_catch()

    orig_img = img_decode(raw_img)

    x_scale, y_scale = img_size(orig_img, des_width=desired_width, mul_acc=True)

    face_num = dir_check()
    face_time = time.time()

    while True:
        try:
            raw_img = stream_catch()
            if raw_img:
                orig_img = img_decode(raw_img)

                # img_face = cv2.imread('./Face1.jpg')
                # img_face = img_size(img_face, des_width=153, grey=False)
                #
                # bHeight = img_face.shape[0]
                # bWidth = img_face.shape[1]
                # x_plus = 40
                # y_plus = 40
                # orig_img[y_plus:y_plus+bHeight, x_plus:x_plus+bWidth, :] = img_face


            img = img_size(orig_img, des_width=desired_width)

            face_roi = face_location(img, x_scale, y_scale)

            names = face_recogn_fun(img, face_roi, x_scale, y_scale)
            print(names)
            name_prints(orig_img, face_roi, names)

            if (time.time() - face_time) > 3 and save_face == '2':
                face_time = time.time()
                if len(names):
                    for name in names:
                        if name != 'Unknown':
                            print('Дверь открыта')
                            driver.find_element('xpath',
                                                '/html/body/div/main/div/main/div[2]/div[1]/main/div/div/div[2]/div[1]').click()

            if time.time() - face_time > 2 and save_face == '1':
                face_num = save_face_box(orig_img, face_roi, face_num)
                face_time = time.time()
            face_box(orig_img, face_roi)

            cv2.imshow("Output", orig_img)
            # cv2.imshow("Output1", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                driver.close()
                driver.quit()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break

        except Exception as E:
            driver.close()
            driver.quit()


def frontal_cam(save_face, person='Strangers', desired_width=700):
    stream = cv2.VideoCapture(0)
    ret, orig_img = stream.read()

    x_scale, y_scale = img_size(orig_img, des_width=desired_width, mul_acc=True)

    face_num = dir_check(person)
    face_time = time.time()

    while True:

        ret, orig_img = stream.read()
        img = img_size(orig_img, des_width=desired_width)

        face_roi = face_location(img, x_scale, y_scale)

        if save_face == '2':
            names = face_recogn_fun(img, face_roi, x_scale, y_scale)
            print(names)
            name_prints(orig_img, face_roi, names)

        if time.time() - face_time > 1 and save_face == '1':
            face_num = save_face_box(img, face_roi, face_num, person, x_scale, y_scale)
            face_time = time.time()
        face_box(orig_img, face_roi)

        cv2.imshow("Output", orig_img)
        # cv2.imshow("Output1", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if save_face == '1': face_train()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            break


if __name__ == '__main__':

    while True:
        camera = input('1-домофон, 2-фронтальная камера 3-выйти\n')
        if camera == '3':
            break
        save_face = input('1-сохранять лица, 2-не сохранять лица\n')

        # Подключаем каскады для определения лиц
        face_cascade = cv2.CascadeClassifier("cascades/face_cv2.xml")
        eye_cascade = cv2.CascadeClassifier("cascades/two_eyes_big.xml")

        # Загрузка файла для распознавания
        try:
            data = pickle.loads(open('face_enc', "rb").read())
        except:
            data = None

        if camera == '1' and (save_face == '1' or save_face == '2'):
            options = Options()
            options.add_argument('--headless')
            driver = webdriver.Chrome('./chromedriver', options=options)
            home_cam(save_face, desired_width=500)

        elif camera == '2' and save_face == '1':
            person = input('Введите имя сохраняемого человека\n\
    или -, если хотите просто сохранить\n')
            if person != '-':
                frontal_cam(save_face, person, desired_width=500)
            else:
                frontal_cam(save_face, desired_width=500)
        elif camera == '2' and save_face == '2':
            frontal_cam(save_face, desired_width=500)
        else:
            print('Введите корректные значения')
