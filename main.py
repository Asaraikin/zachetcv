

'''
img = cv2.imread('images/pencils.jpg')
cv2.imshow('window1', img)
cv2.waitKey(0)


#Захватываем видеофайл
cap = cv2.VideoCapture('video/video_1.mp4')
cap.set(3,300)
cap.set(4,300)
#цикл для отображения ахваченных кадров
while True:
    success, v_img = cap.read()
    v_img = cv2.cvtColor(v_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Window8',v_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#Трансформация изображения
img = cv2.imread('images/pencils.jpg')
print(img.shape)
new_img = cv2.resize(img,(300,300))
print(new_img.shape)
cv2.imshow('window1', new_img)
#cv2.waitKey(100)

#брезка изображения
#cv2.imshow('window3', img[0:100, 0:200])


#Размытие по гауссу
new_img2 = cv2.GaussianBlur(img, (9,9), 5)
cv2.imshow('window4', new_img2)

#ЧБ
new_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('window5', new_img3)
print(new_img3.shape)

#ЧБ битовое изображение
new_img4 = cv2.Canny(new_img3, 10, 10)
cv2.imshow('window6', new_img4)

#Другое изображение
new_img6 = cv2.imread('images/imagee.png')
print(img.shape)
new_img6 = cv2.GaussianBlur(new_img6, (1,1), 95)
cv2.imwrite('images/f1.1.png',new_img6)
cv2.imshow('window4', new_img6)
new_img6 = cv2.cvtColor(new_img6, cv2.COLOR_BGR2GRAY)
cv2.imwrite('images/f2.1.png',new_img6)
new_img6 = cv2.Canny(new_img6, 100, 100)
cv2.imwrite('images/f3.1.png',new_img6)


kernel = np.ones((5,5), np.uint8)

new_img6=cv2.dilate(new_img6, kernel, iterations=1)
new_img6 = cv2.erode(new_img6, kernel, iterations=1)
cv2.imwrite('images/f5.1.png',new_img6)
cv2.imshow('window7',new_img6)
cv2.waitKey(0)
'''
'''
#Создание объектов
#Заполнение нулями области цветного изображения
photo = np.zeros((450,450,3), dtype='uint8')
#Раскрашивание всей области определенным цветом
photo[:] = 255, 200, 255
cv2.imshow('window0', photo)
#Раскрашивание фрагмента области. Первый парамет - высота, второй - ширина
photo[100:150, 200:255] = 255, 224, 255
cv2.imshow('window1', photo)
#Рисование прямоугольника
cv2.rectangle(photo,(10,110),(220,300), (180,0,0), thickness=3)
cv2.imshow('window3',photo)
#Закраска
cv2.rectangle(photo,(10,110),(220,300), (180,0,0), thickness=cv2.FILLED)
cv2.imshow('window4',photo)

'''
#Рисование линии
'''
photo1 = np.zeros((450,450,3), dtype='uint8')
cv2.line(photo1, (150,100), (250,100), (255, 200, 255), thickness=2)
cv2.imshow('window5',photo1)
#Разделение рабочей области попалам
cv2.line(photo1, (0,photo1.shape[0]//2), (photo1.shape[1],photo1.shape[0]//2), (255, 200, 255), thickness=2)
cv2.imshow('window6',photo1)
#рисуем круг
cv2.circle(photo1,(photo1.shape[1]//2, photo1.shape[0]//2),80,(0,0,180),thickness=2)
cv2.imshow('window7',photo1)

cv2.putText(photo1, 'test text', (160,205), cv2.FONT_HERSHEY_TRIPLEX, 1, (222,22,22), 3 )
cv2.imshow('window8',photo1)

court = np.ones((600,1000,3),dtype='uint8')
cv2.rectangle(court,(50,50), (950,550), (0,0,0), 2)
cv2.line(court,(500,50), (500,550), (0,0,0), 2)
cv2.circle(court,(500,300), 80,(0,0,0),2)
cv2.circle(court,(500,300), 5,(0,0,0),1)
cv2.rectangle(court,(50,150), (200,450), (0,0,0), 2)
cv2.rectangle(court,(800,150), (950,450), (0,0,0), 2)

cv2.circle(court,(50,250), (80,350), (0,0,255), 1)
cv2.circle(court,(120,300), 5, (0,0,255), 1)

cv2.circle(court,(920,250), (950,350), (0,0,255), 1)
cv2.circle(court,(800,300), 5, (0,0,255), 1)
cv2.imshow('court_1',court)


cv2.waitKey(0)
cv2.destroyWindow()
'''
''' 
import cv2

img = cv2.imread('images/cat.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('images/cat.jpg', 1)
'''
''' 
cv2.imwrite('images/cat.png', img)
img2 = cv2.imread('images/cat.png', 1)
cv2.imshow('img_gray', img)
cv2.imshow('img_color_original', img1)
cv2.imshow('img_color_from_png', img2)
'''


''' 
import cv2
import numpy as np

# Создание матрицы
n = 40
a = np.ones([n, n])
for i in range(n):
    a[i][i] = 1

for i in range(n):
    for j in range(0, i):
        a[i][j] = 0

# Загрузка изображения
img3 = cv2.imread('images/cat.png', cv2.IMREAD_COLOR)  # Используйте правильный флаг

if img3 is None:
    print("Ошибка: не удалось загрузить изображение 'cat.png'. Проверьте путь.")

# Сохранение матрицы 'a' как изображения (если это нужно)
cv2.imwrite("images/matrix_a.png", a * 255)  # Умножаем на 255, чтобы получить видимое изображение

print(a)

# Отображение загруженного изображения
if img3 is not None:
    cv2.imshow('Image', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
'''
#свойство матрицы изображения
print('серый')
print(type(img))
print(img.shape)
print(img.size)
print(img.dtype)
print('цветной')
print(type(img1))
print(img1.shape)
print(img1.size)
print(img1.dtype)
'''
''' 
#Параметры отдельного пикселя
px1 = img1[600,800]
print(px1)
px = img[600,800]
print(px)

img1[600,800] = [122, 100, 100]
px1 = img1[600,800]
print(px1)

#интенсивность цветогвого канала
blue = img1[600,800,0]
print(blue)
'''
''' 
red = img1.item(600,800,2)
print(red)

img1.itemset((600,800,2), 220)
red1 = img1.item(600,800,2)
print(red1)


# Вывод в одном окне

imgg=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)
plt.subplot(221)
plt.imshow(imgg)
plt.axis('off')
grey_img = cv2.imread('images/cat.jpg', 0)
#бинарноре изображение
im_bw = cv2.threshold(grey_img,128, 255, cv2.THRESH_BINARY)[1]
plt.subplot(222)
plt.imshow(im_bw, 'grey')
plt.axis('off')

#Негатив
im_bwb = cv2.threshold(grey_img,128, 255, cv2.THRESH_BINARY_INV)[1]
plt.subplot(223)
plt.imshow(im_bwb, 'grey')
plt.axis('off')
plt.show()
cv2.waitKey(0)


img = cv2.imread('images/r.jpg')
print(img.shape)
image = cv2.rectangle(img,(270,320),(630,200), (0,0,255), 2)
cv2.imshow('rectangle', img)
f_img = 450
r=float(f_img)/img.shape [1]
dim = (f_img, int(img.shape[0]*r))
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('resiz_img', resized)
print(resized.shape)
print(resized)
cv2.waitKey(0)


 
img = cv2.imread('images/cat.jpg')
kernel = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img,(9,9))
plt.subplot(333), plt.imshow(img), plt.title('orig')
plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(dst), plt.title('aver')
plt.xticks([]), plt.yticks([])
plt.subplot(339), plt.imshow(blur), plt.title('blur')
plt.xticks([]), plt.yticks([])

gauss = cv2.GaussianBlur(img, (5,5), 3)
gauss = cv2.medianBlur(img, 5)
plt.subplot(121), plt.imshow(img), plt.title('Orig')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gauss), plt.title('G_blur')
plt.xticks([]), plt.yticks([])


plt.subplot(121), plt.imshow(img), plt.title('Orig')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gauss), plt.title('G_blur')
plt.xticks([]), plt.yticks([])

plt.show()
'''

'''
#Обнаружение градиентов и перепадов разными методами
img = cv2.imread('Line/6.png', 0)

#функция собеля
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
sobel_hor = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)
cv2.imshow('orig', img)
cv2.imshow('Sobel Vertical', sobel_vertical)
cv2.imshow('Sobel Horizontal', sobel_hor)
cv2.waitKey(0)


# изменение пареметра dtype в функции sobel
sobelx8u = cv2.Sobel(img, cv2.CV_8U , 1, 0 , ksize=5)
sobelx64f = cv2.Sobel(img, cv2.CV_64F , 0, 1 , ksize=5)
abs_sobelx64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobelx64f)
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Orig'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(sobelx8u, cmap='gray'), plt.title('cv2.CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(sobel_8u, cmap='gray'), plt.title('sovel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()


#Обнаружение перепадов метгодом Превита
kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
# маскор координаты у
kernel_y=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx= cv2.filter2D(img, -1, kernel)
img_prewitty=cv2.filter2D(img,-1,kernel_y)
cv2.imshow('kernel_x', img_prewittx)
cv2.imshow('kernel_y', img_prewitty)


#методж РОбертса
kernel1=np.array([[1,0],[1,0]])
kernel2=np.array([[0,1],[0,1]])
img_robx= cv2.filter2D(img,-1,kernel1)
img_roby= cv2.filter2D(img,-1,kernel2)
output_image=img_robx+img_roby
cv2.imshow('output' ,output_image)


#лаплас(поиск резких перепадов и увеличения резкости)
lapla = cv2.Laplacian(img, cv2.CV_64F)
cv2.imshow('Lapla', lapla)

cv2.waitKey(0)


#Выделение контуров
img = cv2.imread('Line/6.png', 0)
imag = cv2.medianBlur(img, 9)
thresh = cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9,2)
cv2.imshow('thresh', thresh)
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt= contours[0]
imag = cv2.drawContours(imag,[cnt], 0, (0,255,0),3)
cv2.imshow('Contours', imag)

#Использование метода Кэнни
edges =cv2.Canny(img,700,100,apertureSize=3)
cv2.imshow('Can_img', edges)

cv2.waitKey(0)
'''

#Морфологические преобразования
#Дилатация(расширение) dilation=cv2.dilate(img,kernel,iteration = 1)
#Эрозия erode = cv2.erode(img,kernel,iteration = 1)


'''
# Чтение изображения
img = cv2.imread('images/R.jpg')
image = Image.open('images/r.jpg')
draw = ImageDraw.Draw(image)

# Получаем размеры изображения
width, height = image.size

# Загружаем изображение для доступа к его пикселям
pix = image.load()

# Итерируем по всем пикселям изображения
for i in range(width):
    for j in range(height):
        rand = random.randint(0, 200)

        # Получаем текущие значения RGB
        r, g, b = pix[i, j]

        # Ограничиваем значения RGB от 0 до 255
        r = min(r + rand, 255)
        g = min(g + rand, 255)
        b = min(b + rand, 255)

        # Устанавливаем новый цвет
        draw.point((i, j), (r, g, b))

# Сохраняем обработанное изображение
image.save('images/RN.png', format='PNG')


img2 = cv2.imread('images/R.jpg', 0)
kernel =np.ones((5,5), np.uint8)
dilation = cv2.dilate(img2, kernel, iterations=1)
erosion = cv2.erode(img2, kernel, iterations=1)
cv2.imshow('dil', dilation)
cv2.imshow('ero', erosion)
cv2.waitKey(0)
'''
'''
img = cv2.imread('images/RN.png', 0)

# Проверка, удалось ли загрузить изображение
if img is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

# Формирование ядра для морфологических операций
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))

# Применение морфологических операций
open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Отображение изображений
cv2.imshow('Original Image', img)
cv2.imshow('Opened Image', open_img)
cv2.imshow('Closed Image', close_img)
'''

'''
import cv2

# Чтение изображения в градациях серого
img = cv2.imread('images/imagee.png', 0)

# Проверка на успешное чтение изображения
if img is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

# Создание ядра для морфологических операций
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Применение морфологических операций
open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Отображение изображений
cv2.imshow('Original Image', img)
cv2.imshow('Opened Image', open_img)
cv2.imshow('Closed Image', close_img)
cv2.imshow('Gradient', gradient)



img = cv2.imread('images/R.jpg')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#Преобразование цилиндра
top=cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
#Преобразование черной шляпы
black=cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('TOP', top)
cv2.imshow('Black', black)
cv2.waitKey(0)
'''
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
ladybug = cv2.imread('images/image.jpg')
ladybug = cv2.cvtColor(ladybug, cv2.COLOR_BGR2RGB)
plt.imshow(ladybug)
plt.show()
# Повышение резкости
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
ladybug = cv2.filter2D(ladybug, -1, kernel)
plt.imshow(ladybug)
plt.show()
hsv_ladybug = cv2.cvtColor(ladybug, cv2.COLOR_RGB2HSV)
# Цветовые границы региона
low_color_hsv = (1, 190, 200)
low_color_rgb = (255, 4, 0)
high_color_hsv = (18, 255, 255)
high_color_rgb = (255, 77, 0)
# Демонстрация двух выбранных границ
low_img = np.zeros((300, 300, 3), np.uint8)
low_img[:] = low_color_rgb
plt.subplot(1, 2, 1)
plt.imshow(low_img)
high_img = np.zeros((300, 300, 3), np.uint8)
high_img[:] = high_color_rgb
plt.subplot(1, 2, 2)
plt.imshow(high_img)
plt.show()
# Границы для белых оттенков
low_white = (0, 0, 200)
high_white = (145, 60, 255)
# Получение двоичной финальной маски
mask_white = cv2.inRange(hsv_ladybug, low_white, high_white)
mask = cv2.inRange(hsv_ladybug, low_color_hsv, high_color_hsv)
final_mask = mask + mask_white
# Применение маски
result = cv2.imread('result.jpg')
result = cv2.bitwise_and(ladybug, ladybug, result, final_mask)
# Маска и исходное изображение с маской сверху
plt.subplot(1, 2, 1)
plt.imshow(final_mask, "gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
# Сглаживание
blur = cv2.GaussianBlur(result, (9, 9), 0)
plt.imshow(blur)
plt.show()
# Поиск краев, основанный на выбранном цветовом регионе
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
result = cv2.drawContours(result, contours, -1, (161, 255, 255), 3, cv2.LINE_AA,hierarchy, 1)
plt.imshow(result)
plt.show()
# Проверка сегментации на другом изображении
bugs = cv2.cvtColor(cv2.imread("images/image2.jpg"), cv2.COLOR_BGR2RGB)
bugs = cv2.filter2D(bugs, -1, kernel)
hsv_bugs = cv2.cvtColor(bugs, cv2.COLOR_RGB2HSV)
mask_white2 = cv2.inRange(hsv_bugs, low_white, high_white)
mask2 = cv2.inRange(hsv_bugs, low_color_hsv, high_color_hsv)
final_mask2 = mask_white2 + mask2
result2 = cv2.imread('result1.jpg')
result2 = cv2.bitwise_and(bugs, bugs, result2, final_mask2)
plt.subplot(1, 2, 1)
plt.imshow(bugs)
plt.subplot(1, 2, 2)
plt.imshow(result2)
plt.show()
blur2 = cv2.GaussianBlur(result2, (9, 9), 0)
contours2, hierarchy2 = cv2.findContours(mask2.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
result2 = cv2.drawContours(result2, contours2, -1, (161, 255, 255), 3,cv2.LINE_AA, hierarchy2, 1)
plt.imshow(result2)
plt.show()
'''



import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('Qt5Agg')
import numpy
from scipy.ndimage import label
img = cv2.imread('images/image.jpg')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, numpy.ones((3, 3),dtype=int),iterations=2)
#thresh1 = cv2.dilate(thresh1,numpy.ones((3, 3), dtype=int),iterations=3)
kernel = np.ones((3,3),dtype=int)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv2.dilate(opening,kernel,iterations=3)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
thresh1 = cv2.subtract(sure_bg,sure_fg)
# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers+1
# markers[thresh1==255] = 0
# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
thresh = cv2.resize(thresh, (0, 0), fx=0.5, fy=0.5)
thresh1 = cv2.resize(thresh1, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("IMG", img)
cv2.imshow("V1", thresh)
cv2.imshow("V1+", thresh1)
cv2.moveWindow("IMG", 0, 290)
cv2.moveWindow("V1", 320, 290)
cv2.moveWindow("V1+", 320, 580)
cv2.waitKey(0)
# v2
def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = cv2.connectedComponents(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255
    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_bin = cv2.threshold(img_gray, 0, 255,cv2.THRESH_OTSU)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,numpy.ones((3, 3), dtype=int))
result = segment_on_dt(img, img_bin)
cv2.imshow("V2", result)
cv2.moveWindow("V2", 640, 290)
result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
cv2.imshow("BORDERS", result)
cv2.moveWindow("BORDERS", 960, 290)
img = cv2.imread('images/image.jpg')
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow("KMeans", res2)
cv2.moveWindow("KMeans", 1280, 290)
cap1 = cv2.VideoCapture('v3.mp4')
while cap1.isOpened():
    ret1, frame1 = cap1.read()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, numpy.ones((3, 3),dtype = int), iterations = 2)
    # thresh = cv2.dilate(thresh, numpy.ones((3, 3), dtype=int), iterations=3)
    # render
    cv2.imshow('ORIGINAL', frame1)  # original
    cv2.imshow('WATERSHED', thresh)
    # move
    frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
    thresh = cv2.resize(thresh, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("ORIGINAL", frame1)
    cv2.imshow("WATERSHED", thresh)
    cv2.imshow("GRAY", gray)
    cv2.moveWindow("ORIGINAL", 0, 0)
    cv2.moveWindow("WATERSHED", 320, 0)
    cv2.moveWindow("GRAY", 640, 0)
    # time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap1.release()
cv2.waitKey(0)
cv2.destroyAllWindows()