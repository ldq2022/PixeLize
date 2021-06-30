# Import Numpy
from skimage.exposure import rescale_intensity
import numpy as np
import cv2

# Import Kivy
import kivy

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.config import Config
from kivy.uix.slider import Slider

Config.set('graphics', 'resizable', True)
kivy.require('1.9.1')








class Main(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2


        no_change = np.array((
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]), dtype="double")

        blur0 = np.array((
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]), dtype="double")

        sharpen = np.array((
        	[0, -2, 0],
        	[-2, 9, -2],
        	[0, -2, 0]), dtype="int")

        sharpen3 = np.array((
        	[0, -3, 0],
        	[-3, 13, -3],
        	[0, -3, 0]), dtype="int")

        sharpen4 = np.array((
        	[0, -4, 0],
        	[-4, 17, -4],
        	[0, -4, 0]), dtype="int")

        sharpen6 = np.array((
        	[0, -6, 0],
        	[-6, 25, -6],
        	[0, -6, 0]), dtype="int")

        sharpen10 = np.array((
        	[0, -10, 0],
        	[-10, 41, -10],
        	[0, -10, 0]), dtype="int")

        sharpen100 = np.array((
        	[0, -100, 0],
        	[-100, 401, -100],
        	[0, -100, 0]), dtype="int")

        direction1 = np.array((
        	[-1, 0, 1],
        	[-1, 0, 1],
        	[-1, 0, 1]), dtype="float")

        direction3 = np.array((
        	[2, 0, -2],
        	[2, 0, -2],
        	[2, 0, -2]), dtype="float")

        direction2 = np.array((
        	[3, 3, 3],
        	[0, 0, 0],
        	[-3, -3, -3]), dtype="float")

        laplacian = np.array((
        	[0, -7, 0],
        	[-7, 28, -7],
        	[0, -7, 0]), dtype="float")




        rgb_array = cv2.imread('grape.jpg')
        convoleOutput = cv2.filter2D(rgb_array, -1, no_change)  # switch no_change with any of above (blur0...)

        #--------------------------------------------------
        # modified_img = imagee.fromarray(convoleOutput, 'RGB')
        # modified_img.save('my.png')
        cv2.imwrite('my.png', convoleOutput)



        # adding image to widget
        self.img = Image(source='my.png')
        self.img.allow_stretch = True
        self.img.keep_ratio = True


        self.ori_img = Image(source='grape.jpg')
        self.ori_img.allow_stretch = True
        self.ori_img.keep_ratio = True

        self.add_widget(Label())
        self.add_widget(Label())

        self.add_widget(self.ori_img)
        self.add_widget(self.img)




        bottom_left = BoxLayout(orientation='horizontal')
        self.add_widget(bottom_left)
        self.slider = Slider(value_track=True, value_track_color=[0, 0, 1, 1])
        self.slider_val = Label(text='Blur Amount:  ' + str(int(self.slider.value)) + '%', font_size=40)

        bottom_left.add_widget(self.slider_val)
        bottom_left.add_widget(self.slider)






        bottom_right = BoxLayout(orientation='vertical')
        self.add_widget(bottom_right)
        bottom_right.add_widget(Label())
        bottom_right.add_widget(Label())
        bottom_right_middle = BoxLayout(orientation='horizontal')
        bottom_right.add_widget(bottom_right_middle)
        bottom_right_middle.add_widget(Label())
        self.apply = Button(text="Apply", font_size=40)
        self.apply.bind(on_press=self.apply_button)
        bottom_right_middle.add_widget(self.apply)
        bottom_right_middle.add_widget(Label())
        bottom_right.add_widget(Label())
        bottom_right.add_widget(Label())





    def apply_button(self, instance):
        # apply image manipulation:

        # blur10 = np.ones((2, 2), dtype="float") * (1.0 / (2 * 2))
        # blur20 = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))
        # blur30 = np.ones((4, 4), dtype="float") * (1.0 / (4 * 4))
        # blur40 = np.ones((5, 5), dtype="float") * (1.0 / (5 * 5))
        # blur50 = np.ones((6, 6), dtype="float") * (1.0 / (6 * 6))
        # blur60 = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
        # blur70 = np.ones((10, 10), dtype="float") * (1.0 / (10 * 10))
        # blur80 = np.ones((14, 14), dtype="float") * (1.0 / (14 * 14))
        # blur90 = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
        # blur100 = np.ones((41, 41), dtype="float") * (1.0 / (41 * 41))


        # if self.slider.value < 10:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur0)
        # elif self.slider.value < 20:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur10)
        # elif self.slider.value < 30:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur20)
        # elif self.slider.value < 40:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur30)
        # elif self.slider.value < 50:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur40)
        # elif self.slider.value < 60:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur50)
        # elif self.slider.value < 70:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur60)
        # elif self.slider.value < 80:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur70)
        # elif self.slider.value < 90:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur80)
        # elif self.slider.value < 100:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur90)
        # else:
        #     convoleOutput = cv2.filter2D(rgb_array, -1, blur100)


        rgb_array = cv2.imread('grape.jpg')

        if self.slider.value < 5:
            blur0 = np.array((
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]), dtype="double")
            convoleOutput = cv2.filter2D(rgb_array, -1, blur0)
        elif self.slider.value <= 10:
            blur10 = np.ones((2, 2), dtype="float") * (1.0 / (2 * 2))
            convoleOutput = cv2.filter2D(rgb_array, -1, blur10)
        else:
            x = int(self.slider.value)
            v = x // 2
            blurX = np.ones((v, v), dtype="int") * (1.0 / (v*v))
            convoleOutput = cv2.filter2D(rgb_array, -1, blurX)



        filename = './new.png'
        status = cv2.imwrite(filename, convoleOutput)
        print("Image written to file-system : ",status)
        self.img.source = filename
        self.img.reload()




    def update(self):
        self.slider_val.text = 'Blur Amount:  ' + str(int(self.slider.value)) + '%'




    def on_touch_up(self, touch):
        self.update()

    def convolve(self, image, kernel):
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        # loop over the input image, "sliding" the kernel across
        # each (x, y)-coordinate from left-to-right and top to
        # bottom
        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                # extract the ROI of the image by extracting the
                # *center* region of the current (x, y)-coordinates
                # dimensions
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                # perform the actual convolution by taking the
                # element-wise multiplicate between the ROI and
                # the kernel, then summing the matrix
                k = (np.multiply(roi, kernel)).sum()
                # store the convolved value in the output (x,y)-
                # coordinate of the output image
                output[y - pad, x - pad] = k



        # rescale the output image to be in the range [0, 255]
        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")
        return output


# creating the App class
class image_modifier(App):

    def build(self):
        return Main()


# run the app
image_modifier().run()
